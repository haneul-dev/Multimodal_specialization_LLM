"""계층형 근거 해석 디코더 - 전체 파이프라인 진입점.

중요: hard 여부로 실행/미실행을 결정하지 않는다. 이 디코더는 항상 실행된다.
hard_signal은 내부 처리 깊이(depth)를 조절하는 파라미터로만 쓰인다.

기존 "Hard Route일 때만 작동" 가정(진행방향 문서 0/1번)을 대체하는 결정이며,
연구2가 넘기는 라우팅 신호의 의미도 이에 맞춰 재정의가 필요하다는 점을 팀과 합의해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .audio_decoder import AudioEvidenceDecoder
from .image_decoder import ImageEvidenceDecoder
from .integrated_decoder import IntegratedEvidence, IntegratedEvidenceDecoder
from .schema import Modality, RawEvidence
from .text_decoder import TextEvidenceDecoder


class Depth(str, Enum):
    LIGHT = "light"  # 근거가 적고 명확 - 이미지/음성 등 고비용 해석은 생략
    FULL = "full"  # 근거가 많거나 모달이 분산 - 1층 전체 실행


def resolve_depth(hard_signal: bool, evidence_count: int) -> Depth:
    """hard 여부(추후 연속 점수로 대체 가능)를 depth로 변환한다.

    실행 여부가 아니라 "얼마나 깊게" 처리할지만 결정한다.
    지금은 hard_signal과 근거 개수만 보는 최소 구현이며,
    모달 분산도 등은 Phase 5에서 반영한다.
    """
    if hard_signal or evidence_count > 5:
        return Depth.FULL
    return Depth.LIGHT


@dataclass
class DecoderInput:
    query: str
    evidences: list[RawEvidence] = field(default_factory=list)
    hard_signal: bool = False  # 연구2가 넘기는 신호. depth 결정에만 사용, on/off 스위치 아님.


class HierarchicalEvidenceDecoder:
    def __init__(self):
        self.text_decoder = TextEvidenceDecoder()
        self.image_decoder = ImageEvidenceDecoder()
        self.audio_decoder = AudioEvidenceDecoder()
        self.integrated_decoder = IntegratedEvidenceDecoder()

    def run(self, decoder_input: DecoderInput) -> IntegratedEvidence:
        depth = resolve_depth(decoder_input.hard_signal, len(decoder_input.evidences))

        # 텍스트 디코더는 depth와 무관하게 항상 실행한다.
        claims = self.text_decoder.decode(decoder_input.query, decoder_input.evidences)

        if depth == Depth.FULL:
            image_evs = [e for e in decoder_input.evidences if e.modality == Modality.IMAGE]
            audio_evs = [e for e in decoder_input.evidences if e.modality == Modality.AUDIO]
            claims += self.image_decoder.decode(decoder_input.query, image_evs)
            claims += self.audio_decoder.decode(decoder_input.query, audio_evs)

        return self.integrated_decoder.integrate(claims)
