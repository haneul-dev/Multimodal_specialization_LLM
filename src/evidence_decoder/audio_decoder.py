"""Audio Evidence Decoder — 1층 (Phase 3 예정).

ASR/음성 이해 모델 연결 전까지는 음성 근거를 건너뛰고 빈 리스트를 반환한다.
파이프라인은 "항상 실행"되어야 하므로 미구현 상태가 전체 흐름을
막아서는 안 된다 — 대신 로그로 스킵 사실을 남긴다.
"""

from __future__ import annotations

import logging

from .llm_client import LLMClient, get_default_client
from .schema import Claim, RawEvidence

logger = logging.getLogger(__name__)


class AudioEvidenceDecoder:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or get_default_client()

    def decode(self, query: str, evidences: list[RawEvidence]) -> list[Claim]:
        if evidences:
            logger.warning(
                "AudioEvidenceDecoder 미구현(Phase 3) — 음성 근거 %d건 스킵",
                len(evidences),
            )
        return []
