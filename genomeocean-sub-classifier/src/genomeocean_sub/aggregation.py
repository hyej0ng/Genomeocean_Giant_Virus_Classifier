"""Aggregate Sub predictions from chunks to contigs and files."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import pandas as pd


SUB_LABELS = {0: "NCLDV", 1: "Mirus"}
PROBABILITY_COLUMNS = {0: "prob_ncldv", 1: "prob_mirus"}
COUNT_COLUMNS = {0: "n_chunks_ncldv", 1: "n_chunks_mirus"}


def _winner(values: Iterable[int], labels: Mapping[int, str]) -> tuple[int, int]:
    counts = Counter(int(value) for value in values)
    invalid = set(counts) - set(labels)
    if invalid:
        raise ValueError(f"Unexpected predicted labels: {sorted(invalid)}")
    winner = min(labels, key=lambda label: (-counts[label], label))
    return winner, counts[winner]


def aggregate_chunks_to_contigs(chunk_results: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "record_id",
        "source_file",
        "contig_id",
        "original_length",
        "clean_length",
        "n_chunks",
        "ignored_tail_bp",
        *COUNT_COLUMNS.values(),
        *PROBABILITY_COLUMNS.values(),
        "ensemble_size",
        "mean_chunk_ensemble_agreement",
        "mean_chunk_confidence_std",
        "predicted_label",
        "predicted_name",
        "predicted_votes",
        "confidence",
    ]
    if chunk_results.empty:
        return pd.DataFrame(columns=output_columns)

    group_columns = [
        "record_id",
        "source_file",
        "contig_id",
        "original_length",
        "clean_length",
        "n_chunks",
        "ignored_tail_bp",
    ]
    rows: list[dict] = []
    for group_key, group in chunk_results.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, group_key))
        winner, votes = _winner(group["predicted_label"], SUB_LABELS)
        counts = Counter(group["predicted_label"].astype(int))
        for label, column in COUNT_COLUMNS.items():
            row[column] = int(counts[label])
        for _, column in PROBABILITY_COLUMNS.items():
            row[column] = float(group[column].mean())
        row["ensemble_size"] = int(
            group["ensemble_size"].max() if "ensemble_size" in group else 1
        )
        row["mean_chunk_ensemble_agreement"] = float(
            group["ensemble_agreement"].mean()
            if "ensemble_agreement" in group
            else 1.0
        )
        row["mean_chunk_confidence_std"] = float(
            group["confidence_std"].mean() if "confidence_std" in group else 0.0
        )
        row["predicted_label"] = winner
        row["predicted_name"] = SUB_LABELS[winner]
        row["predicted_votes"] = votes
        row["confidence"] = row[PROBABILITY_COLUMNS[winner]]
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def aggregate_contigs_to_files(contig_results: pd.DataFrame) -> pd.DataFrame:
    count_columns = {0: "n_contigs_ncldv", 1: "n_contigs_mirus"}
    output_columns = [
        "source_file",
        "n_contigs",
        *count_columns.values(),
        "predicted_label",
        "predicted_name",
        "predicted_votes",
        "confidence",
    ]
    if contig_results.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict] = []
    for source_file, group in contig_results.groupby("source_file", sort=False):
        winner, votes = _winner(group["predicted_label"], SUB_LABELS)
        counts = Counter(group["predicted_label"].astype(int))
        row = {
            "source_file": source_file,
            "n_contigs": len(group),
            "predicted_label": winner,
            "predicted_name": SUB_LABELS[winner],
            "predicted_votes": votes,
            "confidence": float(group.loc[group["predicted_label"] == winner, "confidence"].mean()),
        }
        for label, column in count_columns.items():
            row[column] = int(counts[label])
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)
