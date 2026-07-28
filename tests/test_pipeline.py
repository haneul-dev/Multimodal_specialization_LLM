import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evidence_decoder.pipeline import DecoderInput, Depth, HierarchicalEvidenceDecoder, resolve_depth
from evidence_decoder.schema import Modality, RawEvidence


class TestHierarchicalEvidenceDecoderAlwaysRuns(unittest.TestCase):
    """hard_signal이 False여도(Easy/Medium) 디코더가 항상 실행되는지 검증.

    기존 문서의 "Hard일 때만 작동"을 대체하는 핵심 요구사항이라 별도로 명시적으로 테스트한다.
    """

    def _build_input(self, hard_signal: bool) -> DecoderInput:
        evidences = [
            RawEvidence(source_id="t1", modality=Modality.TEXT, content="핵심 근거 문장", relevance=0.8),
        ]
        return DecoderInput(query="질의", evidences=evidences, hard_signal=hard_signal)

    def test_runs_even_when_hard_signal_is_false(self):
        decoder = HierarchicalEvidenceDecoder()
        result = decoder.run(self._build_input(hard_signal=False))
        self.assertEqual(len(result.claims), 1)

    def test_runs_when_hard_signal_is_true(self):
        decoder = HierarchicalEvidenceDecoder()
        result = decoder.run(self._build_input(hard_signal=True))
        self.assertEqual(len(result.claims), 1)

    def test_hard_signal_only_affects_depth_not_execution(self):
        self.assertEqual(resolve_depth(hard_signal=False, evidence_count=1), Depth.LIGHT)
        self.assertEqual(resolve_depth(hard_signal=True, evidence_count=1), Depth.FULL)


class TestPipelineWithMockDataset(unittest.TestCase):
    def test_end_to_end_with_sample_evidence_json(self):
        path = os.path.join(os.path.dirname(__file__), "..", "mock_data", "sample_evidence.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        evidences = [
            RawEvidence(
                source_id=e["source_id"],
                modality=Modality(e["modality"]),
                content=e["content"],
                relevance=e["relevance"],
            )
            for e in data["evidences"]
        ]
        decoder_input = DecoderInput(query=data["query"], evidences=evidences, hard_signal=data["hard_signal"])

        decoder = HierarchicalEvidenceDecoder()
        result = decoder.run(decoder_input)
        source_ids = [c.source_id for c in result.claims]

        # doc_001과 doc_003은 내용이 같아 중복 제거되어야 함
        self.assertIn("doc_001", source_ids)
        self.assertNotIn("doc_003", source_ids)
        self.assertIn("doc_002", source_ids)
        # 이미지 디코더는 Phase 3 미구현이라 결과에 포함되지 않아야 함
        self.assertNotIn("img_001", source_ids)


if __name__ == "__main__":
    unittest.main()
