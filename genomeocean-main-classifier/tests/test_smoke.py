from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch

from genomeocean_main.aggregation import (
    aggregate_chunks_to_contigs,
    aggregate_contigs_to_files,
)
from genomeocean_main.fasta import read_input_records
from genomeocean_main.predict import (
    MainPredictor,
    ModelSpec,
    ProbabilityEnsembleAccumulator,
    build_model_specs,
)
from genomeocean_main.preprocessing import preprocess_records


class SmokeTests(unittest.TestCase):
    def test_fold_specs_and_probability_mean(self):
        specs = build_model_specs(
            "org/main-model",
            revision="abc123",
            subfolders=["fold1", "fold2", "fold3"],
        )
        self.assertEqual([spec.subfolder for spec in specs], ["fold1", "fold2", "fold3"])
        self.assertEqual(specs[0].revision, "abc123")

        accumulator = ProbabilityEnsembleAccumulator(n_items=2, n_labels=3)
        accumulator.add(torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]]))
        accumulator.add(torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]))
        mean, predicted, votes, agreement, confidence_std = accumulator.finalize()
        self.assertTrue(
            torch.allclose(
                mean,
                torch.tensor(
                    [[0.75, 0.2, 0.05], [0.15, 0.75, 0.1]],
                    dtype=torch.float64,
                ),
            )
        )
        self.assertEqual(predicted.tolist(), [0, 1])
        self.assertEqual(votes.tolist(), [2, 2])
        self.assertEqual(agreement.tolist(), [1.0, 1.0])
        self.assertTrue(torch.all(confidence_std >= 0))

    def test_raw_training_checkpoint_gets_export_hint(self):
        checkpoint = (
            Path(__file__).resolve().parents[4]
            / "finetuning_go_main_add"
            / "ft_models"
            / "train_100M_v1.2_5kb"
            / "fold1"
            / "checkpoint-416820"
        )
        if checkpoint.is_dir():
            with self.assertRaisesRegex(RuntimeError, "prepare_hf_folds.py"):
                MainPredictor._validate_local_model_path(ModelSpec(str(checkpoint)))

    def test_fasta_to_aggregated_result_without_downloading_model(self):
        with TemporaryDirectory() as directory:
            fasta = Path(directory) / "sample.fna"
            fasta.write_text(">contig_1\n" + "ACGT" * 2500 + "\n")

            records = read_input_records(fasta)
            chunks, skipped = preprocess_records(records)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(skipped, [])

            fake_rows = []
            for chunk in chunks:
                row = chunk.to_dict()
                row.pop("sequence")
                row.update(
                    {
                        "predicted_label": 1,
                        "predicted_name": "NCLDV/Mirus",
                        "confidence": 0.8,
                        "prob_cellular": 0.1,
                        "prob_ncldv_mirus": 0.8,
                        "prob_other_viruses": 0.1,
                    }
                )
                fake_rows.append(row)

            contigs = aggregate_chunks_to_contigs(pd.DataFrame(fake_rows))
            files = aggregate_contigs_to_files(contigs)
            self.assertEqual(contigs.loc[0, "predicted_name"], "NCLDV/Mirus")
            self.assertEqual(files.loc[0, "predicted_name"], "NCLDV/Mirus")


if __name__ == "__main__":
    unittest.main()
