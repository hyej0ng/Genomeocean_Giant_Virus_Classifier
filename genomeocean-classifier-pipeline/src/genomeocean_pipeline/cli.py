"""Command-line interface for the Main/Sub five-fold ensemble pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from .pipeline import GenomeOceanPipeline


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping when no path is given."""

    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    value = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Model config must contain a YAML mapping: {config_path}")
    return value


def _model_value(value: Any) -> str | list[str] | None:
    """Accept either one repository/path or a list of fold paths."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value if str(item).strip()]
        return values or None
    text = str(value).strip()
    return text or None


def _subfolder_value(section: dict[str, Any]) -> list[str] | None:
    """Read either ``subfolders`` or the friendlier ``folds`` config key."""

    value = section.get("subfolders", section.get("folds"))
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Model subfolders/folds must be a YAML list")
    values = [str(item).strip() for item in value if str(item).strip()]
    return values or None


def _usable_model_value(value: str | list[str] | None) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_usable_model_value(item) for item in value)
    return bool(value and not value.startswith("your-org/"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genomeocean-classify",
        description=(
            "Run Main and Sub five-fold ensembles to classify FASTA contigs as "
            "Cellular, NCLDV, Mirus, or Other Viruses."
        ),
    )
    parser.add_argument("--input", required=True, help="FASTA file or directory")
    parser.add_argument("--output-dir", required=True, help="directory for outputs")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
        help="YAML file containing model IDs and fold subfolders",
    )
    parser.add_argument(
        "--main-model-id",
        action="append",
        default=None,
        help=(
            "Main Hugging Face ID/local path; repeat for independent fold paths. "
            "For one shared repo, combine with repeated --main-subfolder."
        ),
    )
    parser.add_argument(
        "--sub-model-id",
        action="append",
        default=None,
        help=(
            "Sub Hugging Face ID/local path; repeat for independent fold paths. "
            "For one shared repo, combine with repeated --sub-subfolder."
        ),
    )
    parser.add_argument("--main-subfolder", action="append", default=None)
    parser.add_argument("--sub-subfolder", action="append", default=None)
    parser.add_argument("--main-revision", default=None)
    parser.add_argument("--sub-revision", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1250)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    main_config = config.get("main", {})
    sub_config = config.get("sub", {})
    inference_config = config.get("inference", {})
    if not isinstance(main_config, dict) or not isinstance(sub_config, dict):
        raise SystemExit("The 'main' and 'sub' sections must be YAML mappings")
    if not isinstance(inference_config, dict):
        raise SystemExit("The 'inference' section must be a YAML mapping")

    main_model_id = _model_value(args.main_model_id or main_config.get("model_id"))
    sub_model_id = _model_value(args.sub_model_id or sub_config.get("model_id"))
    if main_model_id is None:
        env_value = os.environ.get("GENOMEOCEAN_MAIN_MODEL")
        main_model_id = _model_value(env_value)
    if sub_model_id is None:
        env_value = os.environ.get("GENOMEOCEAN_SUB_MODEL")
        sub_model_id = _model_value(env_value)
    if not _usable_model_value(main_model_id) or not _usable_model_value(sub_model_id):
        raise SystemExit(
            "Provide real Main/Sub model IDs or paths, or replace the "
            "your-org placeholders in configs/models.yaml"
        )

    # Explicit model IDs are a complete CLI override. In particular, do not
    # accidentally combine five independent paths with the YAML subfolder list.
    main_subfolders = (
        args.main_subfolder
        if args.main_model_id is not None
        else _subfolder_value(main_config)
    )
    sub_subfolders = (
        args.sub_subfolder
        if args.sub_model_id is not None
        else _subfolder_value(sub_config)
    )
    main_revision = args.main_revision or main_config.get("revision")
    sub_revision = args.sub_revision or sub_config.get("revision")
    chunk_size = args.chunk_size or int(inference_config.get("chunk_size", 5000))
    stride = args.stride or int(inference_config.get("stride", 5000))

    pipeline = GenomeOceanPipeline(
        main_model_id,
        sub_model_id,
        main_revision=main_revision,
        sub_revision=sub_revision,
        main_subfolders=main_subfolders,
        sub_subfolders=sub_subfolders,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    result = pipeline.classify(
        args.input,
        args.output_dir,
        chunk_size=chunk_size,
        stride=stride,
    )
    print(
        f"[OK] final predictions for {len(result.final_contigs)} contigs; "
        f"Main ensemble {len(pipeline.main_predictor.model_specs)} models; "
        f"Sub ensemble {len(pipeline.sub_predictor.model_specs)} models; "
        f"outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
