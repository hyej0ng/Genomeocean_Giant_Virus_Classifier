"""Small FASTA reader/writer used by the deployment CLI.

The training repository contains a FASTA generator inside
``prep_go_dataset.py``.  This module keeps only the user-input part and has no
knowledge of labels, folds, or training files.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterable, Iterator, TextIO


FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record plus the input file it came from."""

    source_file: str
    contig_id: str
    description: str
    sequence: str

    @property
    def record_id(self) -> str:
        """Identifier that stays unique when a directory contains many files."""

        return f"{self.source_file}::{self.contig_id}"


def is_fasta_path(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(FASTA_SUFFIXES)


def discover_fasta_files(input_path: str | Path) -> list[Path]:
    """Return one FASTA file or all FASTA files below a directory."""

    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if not is_fasta_path(path):
            raise ValueError(
                f"Unsupported FASTA extension: {path.name}. "
                f"Expected one of: {', '.join(FASTA_SUFFIXES)}"
            )
        return [path]

    files = sorted(candidate for candidate in path.rglob("*") if is_fasta_path(candidate))
    if not files:
        raise ValueError(f"No FASTA files found under: {path}")
    return files


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("rt")


def read_fasta(path: str | Path, *, source_name: str | None = None) -> Iterator[FastaRecord]:
    """Yield records from one FASTA file.

    Duplicate contig IDs are rejected because otherwise downstream results
    cannot be mapped back to a unique input record.
    """

    fasta_path = Path(path)
    source_file = source_name or fasta_path.name
    seen_ids: set[str] = set()
    description: str | None = None
    sequence_parts: list[str] = []

    def build_record() -> FastaRecord:
        assert description is not None
        contig_id = description.split()[0]
        if contig_id in seen_ids:
            raise ValueError(f"Duplicate FASTA ID '{contig_id}' in {fasta_path}")
        seen_ids.add(contig_id)
        return FastaRecord(
            source_file=source_file,
            contig_id=contig_id,
            description=description,
            sequence="".join(sequence_parts),
        )

    with _open_text(fasta_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if description is not None:
                    yield build_record()
                description = line[1:].strip()
                if not description:
                    raise ValueError(f"Empty FASTA header at {fasta_path}:{line_number}")
                sequence_parts = []
            else:
                if description is None:
                    raise ValueError(
                        f"Sequence appears before the first FASTA header at "
                        f"{fasta_path}:{line_number}"
                    )
                sequence_parts.append(line)

    if description is not None:
        yield build_record()


def read_input_records(input_path: str | Path) -> list[FastaRecord]:
    """Read a FASTA file or directory and preserve relative source names."""

    input_root = Path(input_path).expanduser().resolve()
    files = discover_fasta_files(input_root)
    base = input_root if input_root.is_dir() else input_root.parent
    records: list[FastaRecord] = []
    for path in files:
        source_name = path.relative_to(base).as_posix()
        records.extend(read_fasta(path, source_name=source_name))
    if not records:
        raise ValueError(f"No FASTA records found in: {input_root}")
    return records


def write_fasta(
    records: Iterable[FastaRecord],
    output_path: str | Path,
    *,
    use_record_id: bool = False,
    line_width: int = 80,
) -> Path:
    """Write records to FASTA, optionally using globally unique record IDs."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            header = record.record_id if use_record_id else record.description
            handle.write(f">{header}\n")
            for start in range(0, len(record.sequence), line_width):
                handle.write(record.sequence[start : start + line_width] + "\n")
    return path
