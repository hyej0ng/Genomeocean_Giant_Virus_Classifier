import unittest

from genomeocean_sub.fasta import FastaRecord
from genomeocean_sub.preprocessing import clean_dna, make_chunk_spans, preprocess_records


class PreprocessingTests(unittest.TestCase):
    def test_clean_dna_matches_training_behavior(self):
        self.assertEqual(clean_dna("acgtn-RYSW"), "ACGT")

    def test_exact_5kb_produces_one_chunk(self):
        self.assertEqual(make_chunk_spans(5000), [(0, 5000)])

    def test_short_record_is_reported(self):
        record = FastaRecord("candidate.fna", "short", "short", "A" * 4999)
        chunks, skipped = preprocess_records([record])
        self.assertEqual(chunks, [])
        self.assertEqual(skipped[0].reason, "shorter_than_chunk_size")

    def test_main_and_sub_chunk_contract_is_5kb_non_overlapping(self):
        self.assertEqual(
            make_chunk_spans(10000),
            [(0, 5000), (5000, 10000)],
        )


if __name__ == "__main__":
    unittest.main()
