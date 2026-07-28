"""Integrated Evidence Decoder — 2층.

1층 디코더들의 출력(Claim 리스트)을 받아
중복 제거 -> 충돌 표시 -> 관련도 순 정렬 -> 토큰 예산 내 분량 조정을 수행한다.

설계 결정(진행방향 문서 3번 기준):
- 충돌은 직접 판정하지 않고 "A는 X, B는 Y"로 병기해 LLM에 넘긴다 (근거성 우선).
- 가장 관련도 높은 근거를 질의 근처에 배치한다 (lost-in-the-middle 회피).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Claim


@dataclass
class IntegratedEvidence:
    claims: list[Claim]
    conflicts: list[tuple[Claim, Claim]]

    def to_prompt_block(self) -> str:
        lines = [f"[{c.modality.value}:{c.source_id}] {c.claim}" for c in self.claims]
        if self.conflicts:
            lines.append("")
            lines.append("주의 - 상충되는 근거:")
            for a, b in self.conflicts:
                lines.append(f"- {a.source_id}는 '{a.claim}', {b.source_id}는 '{b.claim}'")
        return "\n".join(lines)


def _is_duplicate(a: Claim, b: Claim) -> bool:
    # 완전 일치만 중복으로 취급하는 최소 구현.
    # 유사도 기반 판단은 연구1의 공통 임베딩 공간을 재사용해 Phase 4에서 고도화한다.
    return a.claim.strip() == b.claim.strip()


class IntegratedEvidenceDecoder:
    def __init__(self, token_budget: int = 800):
        self.token_budget = token_budget

    def integrate(self, claims: list[Claim]) -> IntegratedEvidence:
        deduped: list[Claim] = []
        for c in claims:
            if not any(_is_duplicate(c, d) for d in deduped):
                deduped.append(c)

        deduped.sort(key=lambda c: c.relevance, reverse=True)

        trimmed: list[Claim] = []
        used = 0
        for c in deduped:
            cost = len(c.claim)  # 문자 수를 토큰 예산의 근사치로 사용 (실제 토크나이저는 Phase 4에서 교체)
            if used + cost > self.token_budget:
                break
            trimmed.append(c)
            used += cost

        conflicts = self._find_conflicts(trimmed)
        return IntegratedEvidence(claims=trimmed, conflicts=conflicts)

    def _find_conflicts(self, claims: list[Claim]) -> list[tuple[Claim, Claim]]:
        # 실제 모순 탐지(NLI/LLM 판정)는 Phase 4 고도화 대상. 지금은 자리만 잡아둔다.
        return []
