"""근거(Evidence) 공통 스키마.

연구2가 넘기는 검색 결과와 1층 디코더(Text/Image/Audio)의 출력이
모두 이 형식을 따라야 2층 Integrated Evidence Decoder에서 통합할 수 있다.
필드 구성은 "연구3_계층형근거해석디코더_진행방향" 문서의 공통 JSON 스키마를 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class RawEvidence:
    """연구2로부터 넘겨받는, 아직 정제되지 않은 검색 결과 1건."""

    source_id: str
    modality: Modality
    content: str  # 텍스트 원문. 이미지/음성은 파일 경로 또는 사전 캡션.
    relevance: float  # 연구2가 매긴 관련도 점수 (0~1)


@dataclass
class Claim:
    """1층 디코더가 추출한 핵심 근거 1건."""

    claim: str
    modality: Modality
    source_id: str
    relevance: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "modality": self.modality.value,
            "source_id": self.source_id,
            "relevance": self.relevance,
            "confidence": self.confidence,
        }
