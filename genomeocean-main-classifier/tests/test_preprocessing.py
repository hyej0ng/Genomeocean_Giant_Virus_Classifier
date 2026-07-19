import unittest

from genomeocean_main.fasta import FastaRecord
from genomeocean_main.preprocessing import (
    clean_dna,
    make_chunk_spans,
    preprocess_records,
)


class PreprocessingTests(unittest.TestCase):
    def test_clean_dna_matches_training_behavior(self):
        self.assertEqual(clean_dna("acgtn-RYSW"), "ACGT")

    def test_exact_5kb_produces_one_chunk(self):
        self.assertEqual(make_chunk_spans(5000), [(0, 5000)])

    def test_10kb_produces_two_chunks(self):
        self.assertEqual(
            make_chunk_spans(10000),
            [(0, 5000), (5000, 10000)],
        )

    def test_short_record_is_reported_instead_of_silently_disappearing(self):
        record = FastaRecord(
            source_file="example.fna",
            contig_id="short",
            description="short",
            sequence="A" * 4999,
        )
        chunks, skipped = preprocess_records([record])
        self.assertEqual(chunks, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, "shorter_than_chunk_size")

    def test_tail_length_is_recorded(self):
        record = FastaRecord(
            source_file="example.fna",
            contig_id="with_tail",
            description="with_tail",
            sequence="A" * 6100,
        )
        chunks, skipped = preprocess_records([record])
        self.assertEqual(skipped, [])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].ignored_tail_bp, 1100)


if __name__ == "__main__":
    unittest.main()
