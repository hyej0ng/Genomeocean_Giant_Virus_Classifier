"""Command-line interface for the main classifier."""

from __future__ import annotations

import argparse
import os

from .predict import MainPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genomeocean-main",
        description=(
            "Classify FASTA contigs as Cellular, NCLDV/Mirus, or Other Viruses."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="classify a FASTA file or directory")
    predict.add_argument("--input", required=True, help="FASTA file or directory")
    predict.add_argument("--output-dir", required=True, help="directory for TSV outputs")
    predict.add_argument(
        "--model-id",
        action="append",
        default=None,
        help=(
            "Hugging Face model ID or local model path. Repeat this option for "
            "independent fold paths. A single value may also be set with "
            "GENOMEOCEAN_MAIN_MODEL."
        ),
    )
    predict.add_argument(
        "--subfolder",
        action="append",
        default=None,
        help=(
            "Fold subfolder inside one shared --model-id, for example fold1. "
            "Repeat for all ensemble members."
        ),
    )
    predict.add_argument("--revision", default=None, help="model commit/tag to pin")
    predict.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument("--chunk-size", type=int, default=5000)
    predict.add_argument("--stride", type=int, default=5000)
    predict.add_argument("--max-length", type=int, default=1250)
    predict.add_argument("--cache-dir", default=None)
    predict.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not access Hugging Face over the network",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        model_ids = args.model_id
        if not model_ids:
            environment_model = os.environ.get("GENOMEOCEAN_MAIN_MODEL")
            model_ids = [environment_model] if environment_model else None
        if not model_ids:
            raise SystemExit(
                "--model-id is required until a public default model has been published"
            )
        predictor = MainPredictor(
            model_ids,
            revision=args.revision,
            subfolders=args.subfolder,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        bundle = predictor.predict_fasta(
            args.input,
            args.output_dir,
            chunk_size=args.chunk_size,
            stride=args.stride,
        )
        print(
            f"[OK] predicted {len(bundle.contig_results)} contigs; "
            f"ensemble models {len(predictor.model_specs)}; "
            f"skipped {len(bundle.skipped)}; outputs: {args.output_dir}"
        )


if __name__ == "__main__":
    main()
