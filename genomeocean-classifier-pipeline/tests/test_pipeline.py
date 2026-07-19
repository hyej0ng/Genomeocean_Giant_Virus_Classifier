from types import SimpleNamespace
import unittest

import pandas as pd

from genomeocean_main.fasta import FastaRecord
from genomeocean_pipeline.cli import _subfolder_value, _usable_model_value
from genomeocean_pipeline.pipeline import GenomeOceanPipeline


def _main_contig(record, label, name):
    return {
        "record_id": record.record_id,
        "source_file": record.source_file,
        "contig_id": record.contig_id,
        "original_length": len(record.sequence),
        "clean_length": len(record.sequence),
        "n_chunks": 1,
        "ignored_tail_bp": 0,
        "predicted_label": label,
        "predicted_name": name,
        "confidence": 0.9,
    }


class FakeMainPredictor:
    def predict_records(self, records, **kwargs):
        rows = [
            _main_contig(records[0], 0, "Cellular"),
            _main_contig(records[1], 1, "NCLDV/Mirus"),
            _main_contig(records[2], 2, "Other Viruses"),
        ]
        return SimpleNamespace(
            records=list(records),
            chunks=[],
            skipped=[],
            chunk_results=pd.DataFrame(),
            contig_results=pd.DataFrame(rows),
            file_results=pd.DataFrame(),
        )


class FakeSubPredictor:
    def __init__(self):
        self.received_record_ids = []

    def predict_records(self, records, **kwargs):
        self.received_record_ids = [record.record_id for record in records]
        record = records[0]
        row = {
            "record_id": record.record_id,
            "source_file": record.source_file,
            "contig_id": record.contig_id,
            "predicted_label": 1,
            "predicted_name": "Mirus",
            "confidence": 0.8,
        }
        return SimpleNamespace(
            records=list(records),
            chunks=[],
            skipped=[],
            chunk_results=pd.DataFrame(),
            contig_results=pd.DataFrame([row]),
            file_results=pd.DataFrame(),
        )


class PipelineTests(unittest.TestCase):
    def test_model_config_accepts_five_fold_subfolders(self):
        self.assertEqual(
            _subfolder_value({"folds": ["fold1", "fold2", "fold3", "fold4", "fold5"]}),
            ["fold1", "fold2", "fold3", "fold4", "fold5"],
        )
        self.assertTrue(_usable_model_value("org/genomeocean-main"))
        self.assertFalse(_usable_model_value("your-org/genomeocean-main"))

    def test_pipeline_routes_only_main_candidates_to_sub(self):
        records = [
            FastaRecord("sample.fna", "cellular", "cellular", "A" * 5000),
            FastaRecord("sample.fna", "candidate", "candidate", "C" * 5000),
            FastaRecord("sample.fna", "other", "other", "G" * 5000),
        ]
        fake_sub = FakeSubPredictor()
        pipeline = GenomeOceanPipeline(
            main_predictor=FakeMainPredictor(),
            sub_predictor=fake_sub,
        )
        result = pipeline.classify_records(records)

        self.assertEqual(fake_sub.received_record_ids, [records[1].record_id])
        self.assertEqual(
            result.final_contigs["final_name"].tolist(),
            ["Cellular", "Mirus", "Other Viruses"],
        )


if __name__ == "__main__":
    unittest.main()
