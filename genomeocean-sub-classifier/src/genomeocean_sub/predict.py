"""Hugging Face model loading and end-to-end NCLDV/Mirus classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .aggregation import (
    PROBABILITY_COLUMNS,
    SUB_LABELS,
    aggregate_chunks_to_contigs,
    aggregate_contigs_to_files,
)
from .fasta import FastaRecord, read_input_records
from .preprocessing import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_STRIDE,
    SequenceChunk,
    SkippedRecord,
    preprocess_records,
)

ENSEMBLE_METHOD = "mean_probabilities"
SUB_CHUNK_RESULT_COLUMNS = [
    "record_id",
    "source_file",
    "contig_id",
    "original_length",
    "clean_length",
    "n_chunks",
    "ignored_tail_bp",
    "chunk_index",
    "chunk_start",
    "chunk_end",
    "predicted_label",
    "predicted_name",
    "confidence",
    "ensemble_size",
    "ensemble_votes",
    "ensemble_agreement",
    "confidence_std",
    *PROBABILITY_COLUMNS.values(),
]
SKIPPED_RESULT_COLUMNS = [
    "record_id",
    "source_file",
    "contig_id",
    "original_length",
    "clean_length",
    "reason",
]


@dataclass(frozen=True)
class ModelSpec:
    """One Sub ensemble member stored locally or on Hugging Face."""

    model_id: str
    revision: str | None = None
    subfolder: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubPredictionBundle:
    records: list[FastaRecord]
    chunks: list[SequenceChunk]
    skipped: list[SkippedRecord]
    chunk_results: pd.DataFrame
    contig_results: pd.DataFrame
    file_results: pd.DataFrame


def build_model_specs(
    model_ids: str | Sequence[str],
    *,
    revision: str | None = None,
    subfolders: Sequence[str] | None = None,
) -> list[ModelSpec]:
    ids = [model_ids] if isinstance(model_ids, str) else list(model_ids)
    ids = [str(value).strip() for value in ids if str(value).strip()]
    if not ids:
        raise ValueError("At least one model_id is required")

    folders = [str(value).strip() for value in (subfolders or []) if str(value).strip()]
    if folders:
        if len(ids) != 1:
            raise ValueError(
                "subfolders can only be used with one shared model_id; "
                "repeat model_id instead when every fold has a different path"
            )
        return [
            ModelSpec(
                model_id=ids[0],
                revision=revision,
                subfolder=folder,
                name=folder,
            )
            for folder in folders
        ]

    return [
        ModelSpec(
            model_id=model_id,
            revision=revision,
            name=f"model{index}",
        )
        for index, model_id in enumerate(ids, start=1)
    ]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


class ProbabilityEnsembleAccumulator:
    """Accumulate fold outputs and calculate a soft-voting prediction."""

    def __init__(self, n_items: int, n_labels: int):
        self.n_items = n_items
        self.n_labels = n_labels
        self.n_models = 0
        self._sum = torch.zeros((n_items, n_labels), dtype=torch.float64)
        self._sum_squares = torch.zeros((n_items, n_labels), dtype=torch.float64)
        self._votes = torch.zeros((n_items, n_labels), dtype=torch.int64)

    def add(self, probabilities: torch.Tensor) -> None:
        values = probabilities.detach().to(device="cpu", dtype=torch.float64)
        expected = (self.n_items, self.n_labels)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Model probabilities have shape {tuple(values.shape)}; expected {expected}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("Model probabilities contain NaN or infinite values")

        self._sum += values
        self._sum_squares += values.square()
        predicted = values.argmax(dim=1)
        self._votes.scatter_add_(
            1,
            predicted.unsqueeze(1),
            torch.ones((self.n_items, 1), dtype=torch.int64),
        )
        self.n_models += 1

    def finalize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n_models == 0:
            raise ValueError("Cannot finalize an ensemble with no model predictions")

        mean = self._sum / self.n_models
        variance = (self._sum_squares / self.n_models - mean.square()).clamp_min(0)
        standard_deviation = variance.sqrt()
        predicted = mean.argmax(dim=1)
        selected_votes = self._votes.gather(1, predicted.unsqueeze(1)).squeeze(1)
        agreement = selected_votes.to(torch.float64) / self.n_models
        confidence_std = standard_deviation.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        return mean, predicted, selected_votes, agreement, confidence_std


class SubPredictor:
    """Predict NCLDV/Mirus with one model or a sequential fold ensemble."""

    def __init__(
        self,
        model_id: str | Sequence[str],
        *,
        revision: str | None = None,
        subfolders: Sequence[str] | None = None,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1250,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_specs = build_model_specs(
            model_id,
            revision=revision,
            subfolders=subfolders,
        )
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

    def _load_kwargs(self, spec: ModelSpec) -> dict:
        values = {
            "revision": spec.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }
        if spec.subfolder is not None:
            values["subfolder"] = spec.subfolder
        return values

    @staticmethod
    def _validate_local_model_path(spec: ModelSpec) -> None:
        path = Path(spec.model_id).expanduser()
        config_path = path / "config.json"
        if not path.is_dir() or not config_path.is_file():
            return
        try:
            config_text = config_path.read_text()
        except OSError:
            return
        if "DOEJGI/GenomeOcean-100M-v1.2" in config_text:
            raise RuntimeError(
                f"Model checkpoint {path} still references remote DOEJGI custom code. "
                "Run scripts/prepare_hf_folds.py and use the generated fold directory."
            )

    def _predict_one_model(
        self,
        spec: ModelSpec,
        chunks: Sequence[SequenceChunk],
    ) -> torch.Tensor:
        self._validate_local_model_path(spec)
        common = self._load_kwargs(spec)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            model_max_length=self.max_length,
            use_fast=True,
            padding_side="right",
            **common,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.model_id,
            **common,
        )
        model.to(self.device)
        model.eval()

        batches: list[torch.Tensor] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            encoded = tokenizer(
                [chunk.sequence for chunk in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            encoded.pop("token_type_ids", None)
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            if probabilities.shape[1] != len(SUB_LABELS):
                raise RuntimeError(
                    f"Sub model '{spec.name}' returned {probabilities.shape[1]} labels; "
                    f"expected {len(SUB_LABELS)}"
                )
            batches.append(probabilities)
        return torch.cat(batches, dim=0)

    def _release_device_cache(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def predict_chunks(self, chunks: Sequence[SequenceChunk]) -> pd.DataFrame:
        if not chunks:
            return pd.DataFrame(columns=SUB_CHUNK_RESULT_COLUMNS)

        accumulator = ProbabilityEnsembleAccumulator(
            n_items=len(chunks),
            n_labels=len(SUB_LABELS),
        )
        for spec in self.model_specs:
            try:
                probabilities = self._predict_one_model(spec, chunks)
                accumulator.add(probabilities)
                del probabilities
            finally:
                self._release_device_cache()

        mean, predicted, votes, agreement, confidence_std = accumulator.finalize()
        rows: list[dict] = []
        for index, chunk in enumerate(chunks):
            label = int(predicted[index].item())
            probability_row = mean[index]
            row = chunk.to_dict()
            row.pop("sequence")
            row["predicted_label"] = label
            row["predicted_name"] = SUB_LABELS[label]
            row["confidence"] = float(probability_row[label].item())
            row["ensemble_size"] = accumulator.n_models
            row["ensemble_votes"] = int(votes[index].item())
            row["ensemble_agreement"] = float(agreement[index].item())
            row["confidence_std"] = float(confidence_std[index].item())
            for label_index, column in PROBABILITY_COLUMNS.items():
                row[column] = float(probability_row[label_index].item())
            rows.append(row)
        return pd.DataFrame(rows, columns=SUB_CHUNK_RESULT_COLUMNS)

    def predict_records(
        self,
        records: Sequence[FastaRecord],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> SubPredictionBundle:
        chunks, skipped = preprocess_records(
            records,
            chunk_size=chunk_size,
            stride=stride,
        )
        chunk_results = self.predict_chunks(chunks)
        contig_results = aggregate_chunks_to_contigs(chunk_results)
        return SubPredictionBundle(
            records=list(records),
            chunks=chunks,
            skipped=skipped,
            chunk_results=chunk_results,
            contig_results=contig_results,
            file_results=aggregate_contigs_to_files(contig_results),
        )

    def predict_fasta(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stride: int = DEFAULT_STRIDE,
    ) -> SubPredictionBundle:
        records = read_input_records(input_path)
        bundle = self.predict_records(records, chunk_size=chunk_size, stride=stride)
        write_sub_outputs(
            bundle,
            output_dir,
            metadata={
                "input": str(Path(input_path).expanduser().resolve()),
                "ensemble_method": ENSEMBLE_METHOD,
                "ensemble_size": len(self.model_specs),
                "models": [spec.to_dict() for spec in self.model_specs],
                "sequential_model_loading": True,
                "device": str(self.device),
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "chunk_size": chunk_size,
                "stride": stride,
            },
        )
        return bundle


def write_sub_outputs(
    bundle: SubPredictionBundle,
    output_dir: str | Path,
    *,
    metadata: dict | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "chunks": out / "chunk_predictions.tsv",
        "contigs": out / "contig_predictions.tsv",
        "files": out / "file_predictions.tsv",
        "skipped": out / "skipped_records.tsv",
        "metadata": out / "run_metadata.json",
    }
    bundle.chunk_results.reindex(columns=SUB_CHUNK_RESULT_COLUMNS).to_csv(
        paths["chunks"], sep="\t", index=False
    )
    bundle.contig_results.to_csv(paths["contigs"], sep="\t", index=False)
    bundle.file_results.to_csv(paths["files"], sep="\t", index=False)
    pd.DataFrame(
        [item.to_dict() for item in bundle.skipped],
        columns=SKIPPED_RESULT_COLUMNS,
    ).to_csv(
        paths["skipped"],
        sep="\t",
        index=False,
    )
    run_metadata = {
        **(metadata or {}),
        "records": len(bundle.records),
        "chunks": len(bundle.chunks),
        "predicted_contigs": len(bundle.contig_results),
        "skipped_records": len(bundle.skipped),
        "outputs": {name: str(path) for name, path in paths.items() if name != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(run_metadata, indent=2) + "\n")
    return paths
