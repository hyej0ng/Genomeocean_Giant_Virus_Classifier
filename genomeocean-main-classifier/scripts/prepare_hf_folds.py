"""Prepare inference-only fold folders for a Hugging Face model repository.

The training checkpoints contain ``modeling_mistral.py`` but their config points
to the original DOEJGI repository for ``configuration_mistral.py``. This script
copies the required inference files and rewrites that reference to local files.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil


def _configuration_source() -> Path:
    module = importlib.import_module("transformers.models.mistral.configuration_mistral")
    return Path(module.__file__).resolve()


def _write_configuration_source(destination: Path) -> None:
    # The installed Transformers file uses package-relative imports because it
    # normally lives under transformers.models.mistral. A Hub custom-code file
    # is loaded from a different package, so use absolute imports there.
    text = _configuration_source().read_text()
    text = text.replace("from ...configuration_utils", "from transformers.configuration_utils")
    text = text.replace("from ...utils", "from transformers.utils")
    destination.write_text(text)


def _copy_model_files(source: Path, destination: Path) -> list[str]:
    required = {"config.json", "modeling_mistral.py"}
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(f"{source / name} is required in the checkpoint")

    copied: list[str] = []
    for path in source.iterdir():
        name = path.name
        is_inference_file = (
            name in required
            or name.startswith("tokenizer")
            or name in {"special_tokens_map.json", "added_tokens.json"}
            or name.endswith((".safetensors", ".bin"))
            or name.endswith(".safetensors.index.json")
            or name.endswith(".bin.index.json")
        )
        if is_inference_file and path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(name)
    if not any(name.endswith((".safetensors", ".bin")) for name in copied):
        raise FileNotFoundError(f"No model weight file found in {source}")
    _write_configuration_source(destination / "configuration_mistral.py")
    copied.append("configuration_mistral.py")
    return sorted(copied)


def prepare_checkpoint(checkpoint: Path, destination: Path) -> list[str]:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output fold directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = _copy_model_files(checkpoint, destination)

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    auto_map = dict(config.get("auto_map", {}))
    auto_map["AutoConfig"] = "configuration_mistral.MistralConfig"
    for auto_name in (
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForMaskedLM",
        "AutoModelForSequenceClassification",
    ):
        if auto_name in auto_map:
            auto_map[auto_name] = f"modeling_mistral.{auto_map[auto_name].split('.')[-1]}"
    config["auto_map"] = auto_map
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create self-contained fold folders for Hugging Face upload."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint directory; repeat once for each fold in fold order.",
    )
    parser.add_argument("--output-dir", required=True, help="New model repository directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, value in enumerate(args.checkpoint, start=1):
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        destination = output_dir / f"fold{index}"
        copied = prepare_checkpoint(checkpoint, destination)
        manifest.append({"fold": f"fold{index}", "source": str(checkpoint), "files": copied})
    (output_dir / "ensemble_manifest.json").write_text(
        json.dumps({"folds": manifest, "n_folds": len(manifest)}, indent=2) + "\n"
    )
    print(f"[OK] prepared {len(manifest)} folds under {output_dir}")


if __name__ == "__main__":
    main()
