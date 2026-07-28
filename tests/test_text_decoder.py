import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evidence_decoder.schema import Modality, RawEvidence
from evidence_decoder.text_decoder import TextEvidenceDecoder


class TestTextEvidenceDecoder(unittest.TestCase):
    def test_decode_extracts_claim_per_text_evidence_and_skips_other_modalities(self):
        decoder = TextEvidenceDecoder()
        evidences = [
            RawEvidence(source_id="a", modality=Modality.TEXT, content="핵심 문장입니다.", relevance=0.9),
            RawEvidence(source_id="b", modality=Modality.IMAGE, content="이미지 설명", relevance=0.5),
        ]

        claims = decoder.decode("질의", evidences)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].source_id, "a")
        self.assertIn("핵심 문장입니다.", claims[0].claim)


if __name__ == "__main__":
    unittest.main()
