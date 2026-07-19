import unittest
from pathlib import Path

import pandas as pd
import torch

from genomeocean_sub.aggregation import (
    aggregate_chunks_to_contigs,
    aggregate_contigs_to_files,
)
from genomeocean_sub.fasta import FastaRecord
from genomeocean_sub.predict import (
    ModelSpec,
    ProbabilityEnsembleAccumulator,
    SubPredictor,
    build_model_specs,
)
from genomeocean_sub.preprocessing import preprocess_records


class SmokeTests(unittest.TestCase):
    def test_fold_specs_and_probability_mean(self):
        specs = build_model_specs(
            ["/models/sub-fold1", "/models/sub-fold2"],
        )
        self.assertEqual([spec.model_id for spec in specs], ["/models/sub-fold1", "/models/sub-fold2"])
        self.assertEqual([spec.name for spec in specs], ["model1", "model2"])

        accumulator = ProbabilityEnsembleAccumulator(n_items=1, n_labels=2)
        accumulator.add(torch.tensor([[0.8, 0.2]]))
        accumulator.add(torch.tensor([[0.4, 0.6]]))
        mean, predicted, votes, agreement, _ = accumulator.finalize()
        self.assertTrue(
            torch.allclose(mean, torch.tensor([[0.6, 0.4]], dtype=torch.float64))
        )
        self.assertEqual(predicted.tolist(), [0])
        self.assertEqual(votes.tolist(), [1])
        self.assertEqual(agreement.tolist(), [0.5])

    def test_raw_training_checkpoint_gets_export_hint(self):
        checkpoint = (
            Path(__file__).resolve().parents[4]
            / "finetuning_go_sub_add"
            / "ft_models"
            / "train_100M_v1.2_5kb"
            / "fold1"
            / "checkpoint-44240"
        )
        if checkpoint.is_dir():
            with self.assertRaisesRegex(RuntimeError, "prepare_hf_folds.py"):
                SubPredictor._validate_local_model_path(ModelSpec(str(checkpoint)))

    def test_fake_mirus_predictions_aggregate_to_mirus(self):
        record = FastaRecord(
            "candidates.fna",
            "candidate_1",
            "candidate_1",
            "A" * 10000,
        )
        chunks, skipped = preprocess_records([record])
        self.assertEqual(skipped, [])

        rows = []
        for chunk in chunks:
            row = chunk.to_dict()
            row.pop("sequence")
            row.update(
                {
                    "predicted_label": 1,
                    "predicted_name": "Mirus",
                    "confidence": 0.9,
                    "prob_ncldv": 0.1,
                    "prob_mirus": 0.9,
                }
            )
            rows.append(row)
        contigs = aggregate_chunks_to_contigs(pd.DataFrame(rows))
        files = aggregate_contigs_to_files(contigs)
        self.assertEqual(contigs.loc[0, "predicted_name"], "Mirus")
        self.assertEqual(files.loc[0, "predicted_name"], "Mirus")


if __name__ == "__main__":
    unittest.main()
