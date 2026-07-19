"""Connect the Main and Sub classifiers without duplicating their model code."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from genomeocean_main.fasta import FastaRecord, read_input_records
from genomeocean_main.predict import MainPredictor, write_main_outputs
from genomeocean_sub.aggregation import (
    aggregate_chunks_to_contigs as aggregate_sub_chunks,
    aggregate_contigs_to_files as aggregate_sub_contigs,
)
from genomeocean_sub.predict import SubPredictionBundle, SubPredictor, write_sub_outputs


FINAL_LABELS = {
    0: "Cellular",
    1: "NCLDV",
    2: "Mirus",
    3: "Other Viruses",
    -1: "Unresolved NCLDV/Mirus",
}
FINAL_LABEL_ORDER = (0, 1, 2, 3, -1)


@dataclass
class PipelineResult:
    records: list[FastaRecord]
    main_bundle: object
    sub_bundle: object
    final_contigs: pd.DataFrame
    final_files: pd.DataFrame


def _empty_sub_bundle() -> SubPredictionBundle:
    empty = pd.DataFrame()
    contigs = aggregate_sub_chunks(empty)
    return SubPredictionBundle(
        records=[],
        chunks=[],
        skipped=[],
        chunk_results=empty,
        contig_results=contigs,
        file_results=aggregate_sub_contigs(contigs),
    )


def combine_contig_results(
    main_contigs: pd.DataFrame,
    sub_contigs: pd.DataFrame,
) -> pd.DataFrame:
    """Map the hierarchical 3-class + 2-class output to final four classes."""

    output_columns = [
        "record_id",
        "source_file",
        "contig_id",
        "original_length",
        "clean_length",
        "n_chunks",
        "ignored_tail_bp",
        "main_label",
        "main_name",
        "main_confidence",
        "main_ensemble_size",
        "main_mean_chunk_ensemble_agreement",
        "main_mean_chunk_confidence_std",
        "sub_label",
        "sub_name",
        "sub_confidence",
        "sub_ensemble_size",
        "sub_mean_chunk_ensemble_agreement",
        "sub_mean_chunk_confidence_std",
        "final_label",
        "final_name",
        "final_confidence",
    ]
    if main_contigs.empty:
        return pd.DataFrame(columns=output_columns)

    sub_lookup = (
        sub_contigs.set_index("record_id", drop=False).to_dict("index")
        if not sub_contigs.empty
        else {}
    )
    rows: list[dict] = []
    for _, main_row in main_contigs.iterrows():
        main_label = int(main_row["predicted_label"])
        sub_row = sub_lookup.get(main_row["record_id"])
        sub_label = None
        sub_name = None
        sub_confidence = None
        sub_ensemble_size = None
        sub_mean_agreement = None
        sub_mean_confidence_std = None

        if main_label == 0:
            final_label = 0
            final_confidence = float(main_row["confidence"])
        elif main_label == 2:
            final_label = 3
            final_confidence = float(main_row["confidence"])
        elif main_label == 1 and sub_row is not None:
            sub_label = int(sub_row["predicted_label"])
            sub_name = str(sub_row["predicted_name"])
            sub_confidence = float(sub_row["confidence"])
            sub_ensemble_size = int(sub_row.get("ensemble_size", 1))
            sub_mean_agreement = float(
                sub_row.get("mean_chunk_ensemble_agreement", 1.0)
            )
            sub_mean_confidence_std = float(
                sub_row.get("mean_chunk_confidence_std", 0.0)
            )
            final_label = 1 if sub_label == 0 else 2
            # Hierarchical confidence: confidence of reaching the Main branch
            # multiplied by confidence within the Sub branch.
            final_confidence = float(main_row["confidence"]) * sub_confidence
        else:
            final_label = -1
            final_confidence = float(main_row["confidence"])

        rows.append(
            {
                "record_id": main_row["record_id"],
                "source_file": main_row["source_file"],
                "contig_id": main_row["contig_id"],
                "original_length": int(main_row["original_length"]),
                "clean_length": int(main_row["clean_length"]),
                "n_chunks": int(main_row["n_chunks"]),
                "ignored_tail_bp": int(main_row["ignored_tail_bp"]),
                "main_label": main_label,
                "main_name": main_row["predicted_name"],
                "main_confidence": float(main_row["confidence"]),
                "main_ensemble_size": int(main_row.get("ensemble_size", 1)),
                "main_mean_chunk_ensemble_agreement": float(
                    main_row.get("mean_chunk_ensemble_agreement", 1.0)
                ),
                "main_mean_chunk_confidence_std": float(
                    main_row.get("mean_chunk_confidence_std", 0.0)
                ),
                "sub_label": sub_label,
                "sub_name": sub_name,
                "sub_confidence": sub_confidence,
                "sub_ensemble_size": sub_ensemble_size,
                "sub_mean_chunk_ensemble_agreement": sub_mean_agreement,
                "sub_mean_chunk_confidence_std": sub_mean_confidence_std,
                "final_label": final_label,
                "final_name": FINAL_LABELS[final_label],
                "final_confidence": final_confidence,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def _final_winner(values: Sequence[int]) -> tuple[int, int]:
    counts = Counter(int(value) for value in values)
    rank = {label: index for index, label in enumerate(FINAL_LABEL_ORDER)}
    winner = min(counts, key=lambda label: (-counts[label], rank.get(label, 999)))
    return winner, counts[winner]


def aggregate_final_files(final_contigs: pd.DataFrame) -> pd.DataFrame:
    count_columns = {
        0: "n_contigs_cellular",
        1: "n_contigs_ncldv",
        2: "n_contigs_mirus",
        3: "n_contigs_other_viruses",
        -1: "n_contigs_unresolved",
    }
    output_columns = [
        "source_file",
        "n_contigs",
        *count_columns.values(),
        "final_label",
        "final_name",
        "final_votes",
        "final_confidence",
    ]
    if final_contigs.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict] = []
    for source_file, group in final_contigs.groupby("source_file", sort=False):
        winner, votes = _final_winner(group["final_label"].astype(int).tolist())
        counts = Counter(group["final_label"].astype(int))
        row = {
            "source_file": source_file,
            "n_contigs": len(group),
            "final_label": winner,
            "final_name": FINAL_LABELS[winner],
            "final_votes": votes,
            "final_confidence": float(
                group.loc[group["final_label"] == winner, "final_confidence"].mean()
            ),
        }
        for label, column in count_columns.items():
            row[column] = int(counts[label])
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


class GenomeOceanPipeline:
    """Run Main first, then run Sub only on the NCLDV/Mirus branch."""

    def __init__(
        self,
        main_model_id: str | Sequence[str] | None = None,
        sub_model_id: str | Sequence[str] | None = None,
        *,
        main_revision: str | None = None,
        sub_revision: str | None = None,
        main_subfolders: Sequence[str] | None = None,
        sub_subfolders: Sequence[str] | None = None,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1250,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        main_predictor: object | None = None,
        sub_predictor: object | None = None,
    ):
        # Predictor injection keeps unit tests fast and avoids downloading a
        # 450 MB model in every CI run.
        self.main_predictor = main_predictor or MainPredictor(
            main_model_id or "",
            revision=main_revision,
            subfolders=main_subfolders,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.sub_predictor = sub_predictor or SubPredictor(
            sub_model_id or "",
            revision=sub_revision,
            subfolders=sub_subfolders,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.main_model_id = main_model_id
        self.sub_model_id = sub_model_id
        self.main_revision = main_revision
        self.sub_revision = sub_revision
        self.main_subfolders = list(main_subfolders or [])
        self.sub_subfolders = list(sub_subfolders or [])
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def classify_records(
        self,
        records: Sequence[FastaRecord],
        *,
        chunk_size: int = 5000,
        stride: int = 5000,
    ) -> PipelineResult:
        main_bundle = self.main_predictor.predict_records(
            records,
            chunk_size=chunk_size,
            stride=stride,
        )
        candidate_ids = set(
            main_bundle.contig_results.loc[
                main_bundle.contig_results["predicted_label"] == 1,
                "record_id",
            ]
        )
        candidate_records = [
            record for record in records if record.record_id in candidate_ids
        ]
        if candidate_records:
            sub_bundle = self.sub_predictor.predict_records(
                candidate_records,
                chunk_size=chunk_size,
                stride=stride,
            )
        else:
            sub_bundle = _empty_sub_bundle()

        final_contigs = combine_contig_results(
            main_bundle.contig_results,
            sub_bundle.contig_results,
        )
        return PipelineResult(
            records=list(records),
            main_bundle=main_bundle,
            sub_bundle=sub_bundle,
            final_contigs=final_contigs,
            final_files=aggregate_final_files(final_contigs),
        )

    def classify(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        chunk_size: int = 5000,
        stride: int = 5000,
    ) -> PipelineResult:
        records = read_input_records(input_path)
        result = self.classify_records(
            records,
            chunk_size=chunk_size,
            stride=stride,
        )
        write_pipeline_outputs(
            result,
            output_dir,
            metadata={
                "input": str(Path(input_path).expanduser().resolve()),
                "ensemble_method": "mean_probabilities",
                "main_models": [
                    spec.to_dict()
                    for spec in getattr(self.main_predictor, "model_specs", [])
                ],
                "sub_models": [
                    spec.to_dict()
                    for spec in getattr(self.sub_predictor, "model_specs", [])
                ],
                "main_ensemble_size": len(
                    getattr(self.main_predictor, "model_specs", [])
                ),
                "sub_ensemble_size": len(
                    getattr(self.sub_predictor, "model_specs", [])
                ),
                "sequential_model_loading": True,
                "device": self.device,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "chunk_size": chunk_size,
                "stride": stride,
            },
        )
        return result


def write_pipeline_outputs(
    result: PipelineResult,
    output_dir: str | Path,
    *,
    metadata: dict | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_main_outputs(result.main_bundle, out / "main", metadata=metadata)
    write_sub_outputs(result.sub_bundle, out / "sub", metadata=metadata)

    paths = {
        "final_contigs": out / "final_contig_predictions.tsv",
        "final_files": out / "final_file_predictions.tsv",
        "metadata": out / "run_metadata.json",
    }
    result.final_contigs.to_csv(paths["final_contigs"], sep="\t", index=False)
    result.final_files.to_csv(paths["final_files"], sep="\t", index=False)
    run_metadata = {
        **(metadata or {}),
        "records": len(result.records),
        "main_predicted_contigs": len(result.main_bundle.contig_results),
        "sub_candidate_contigs": len(result.sub_bundle.contig_results),
        "final_predicted_contigs": len(result.final_contigs),
        "unresolved_contigs": int(
            (result.final_contigs["final_label"] == -1).sum()
            if not result.final_contigs.empty
            else 0
        ),
        "outputs": {
            "main": str(out / "main"),
            "sub": str(out / "sub"),
            "final_contigs": str(paths["final_contigs"]),
            "final_files": str(paths["final_files"]),
        },
    }
    paths["metadata"].write_text(json.dumps(run_metadata, indent=2) + "\n")
    return paths
