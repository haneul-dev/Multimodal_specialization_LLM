import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evidence_decoder.integrated_decoder import IntegratedEvidenceDecoder
from evidence_decoder.schema import Claim, Modality


class TestIntegratedEvidenceDecoder(unittest.TestCase):
    def test_dedup_and_sort_by_relevance(self):
        claims = [
            Claim(claim="A", modality=Modality.TEXT, source_id="1", relevance=0.5, confidence=1.0),
            Claim(claim="B", modality=Modality.TEXT, source_id="2", relevance=0.9, confidence=1.0),
            Claim(claim="A", modality=Modality.TEXT, source_id="3", relevance=0.99, confidence=1.0),
        ]
        decoder = IntegratedEvidenceDecoder(token_budget=1000)

        result = decoder.integrate(claims)

        self.assertEqual([c.claim for c in result.claims], ["B", "A"])

    def test_token_budget_trims_low_relevance_claims(self):
        claims = [
            Claim(claim="x" * 10, modality=Modality.TEXT, source_id="1", relevance=0.9, confidence=1.0),
            Claim(claim="y" * 10, modality=Modality.TEXT, source_id="2", relevance=0.1, confidence=1.0),
        ]
        decoder = IntegratedEvidenceDecoder(token_budget=10)

        result = decoder.integrate(claims)

        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.claims[0].source_id, "1")


if __name__ == "__main__":
    unittest.main()
