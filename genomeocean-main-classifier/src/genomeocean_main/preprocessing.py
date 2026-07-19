"""Training-compatible DNA cleaning and fixed-size chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from .fasta import FastaRecord


DEFAULT_CHUNK_SIZE = 5000
DEFAULT_STRIDE = 5000


@dataclass(frozen=True)
class SequenceChunk:
    record_id: str
    source_file: str
    contig_id: str
    original_length: int
    clean_length: int
    n_chunks: int
    ignored_tail_bp: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    sequence: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SkippedRecord:
    record_id: str
    source_file: str
    contig_id: str
    original_length: int
    clean_length: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_dna(sequence: str) -> str:
    """Match the original training preprocessing exactly.

    Only A/C/G/T/N are retained first, then N is removed.  Other IUPAC
    ambiguity symbols are removed as they were in ``prep_go_dataset.py``.
    """

    cleaned = re.sub(r"[^ACGTNacgtn]", "", sequence).upper()
    return cleaned.replace("N", "")


def make_chunk_spans(
    sequence_length: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> list[tuple[int, int]]:
    """Return spans for complete chunks; a shorter final tail is not emitted."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if sequence_length < chunk_size:
        return []
    return [
        (start, start + chunk_size)
        for start in range(0, sequence_length - chunk_size + 1, stride)
    ]


def preprocess_records(
    records: Iterable[FastaRecord],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[list[SequenceChunk], list[SkippedRecord]]:
    """Clean records and split them into complete chunks."""

    chunks: list[SequenceChunk] = []
    skipped: list[SkippedRecord] = []

    for record in records:
        cleaned = clean_dna(record.sequence)
        spans = make_chunk_spans(
            len(cleaned),
            chunk_size=chunk_size,
            stride=stride,
        )
        if not spans:
            reason = "empty_after_cleaning" if not cleaned else "shorter_than_chunk_size"
            skipped.append(
                SkippedRecord(
                    record_id=record.record_id,
                    source_file=record.source_file,
                    contig_id=record.contig_id,
                    original_length=len(record.sequence),
                    clean_length=len(cleaned),
                    reason=reason,
                )
            )
            continue

        last_covered_end = max(end for _, end in spans)
        ignored_tail_bp = max(0, len(cleaned) - last_covered_end)
        for chunk_index, (start, end) in enumerate(spans):
            chunks.append(
                SequenceChunk(
                    record_id=record.record_id,
                    source_file=record.source_file,
                    contig_id=record.contig_id,
                    original_length=len(record.sequence),
                    clean_length=len(cleaned),
                    n_chunks=len(spans),
                    ignored_tail_bp=ignored_tail_bp,
                    chunk_index=chunk_index,
                    chunk_start=start,
                    chunk_end=end,
                    sequence=cleaned[start:end],
                )
            )

    return chunks, skipped
