"""Training-compatible DNA cleaning and fixed-size chunking for Sub."""

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
    cleaned = re.sub(r"[^ACGTNacgtn]", "", sequence).upper()
    return cleaned.replace("N", "")


def make_chunk_spans(
    sequence_length: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> list[tuple[int, int]]:
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
            skipped.append(
                SkippedRecord(
                    record_id=record.record_id,
                    source_file=record.source_file,
                    contig_id=record.contig_id,
                    original_length=len(record.sequence),
                    clean_length=len(cleaned),
                    reason=(
                        "empty_after_cleaning"
                        if not cleaned
                        else "shorter_than_chunk_size"
                    ),
                )
            )
            continue

        ignored_tail_bp = max(0, len(cleaned) - max(end for _, end in spans))
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
