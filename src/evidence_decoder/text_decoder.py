"""Text Evidence Decoder — 1층.

검색된 텍스트 근거에서 질의와 관련된 핵심 문장(claim)을 추출한다.
"추출형 vs 생성형" 설계 결정에 따라 원문 그대로 뽑아내는 추출형을 기본으로 한다
(환각을 줄이기 위함, 진행방향 문서 3번).
"""

from __future__ import annotations

from .llm_client import LLMClient, get_default_client
from .schema import Claim, Modality, RawEvidence

_PROMPT_TEMPLATE = """다음은 질의와 검색된 텍스트 근거입니다.
질의와 직접 관련된 핵심 문장만 원문 그대로 1~2개 추출하세요. 새로 요약하지 말고 원문에서 그대로 가져오세요.

QUERY: {query}
CONTENT:
{content}
"""


class TextEvidenceDecoder:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or get_default_client()

    def decode(self, query: str, evidences: list[RawEvidence]) -> list[Claim]:
        claims: list[Claim] = []
        for ev in evidences:
            if ev.modality != Modality.TEXT:
                continue
            prompt = _PROMPT_TEMPLATE.format(query=query, content=ev.content)
            extracted = self.llm_client.extract(prompt).strip()
            if not extracted:
                continue
            claims.append(
                Claim(
                    claim=extracted,
                    modality=Modality.TEXT,
                    source_id=ev.source_id,
                    relevance=ev.relevance,
                    confidence=1.0,  # mock 단계 기본값. 실제 모델 연결 후 응답 기반으로 보정.
                )
            )
        return claims
