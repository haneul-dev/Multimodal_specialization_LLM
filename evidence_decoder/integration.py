"""2층 - 근거 통합 계층.

모달리티별 카드를 하나의 근거 집합으로 합친다.
중복제거 / 충돌확인 / 우선순위 / 분량최적화 네 가지가 목적이다.

속도가 이 연구의 핵심 지표이므로 2단 구조로 만든다.
1. 규칙 기반 사전정리 - LLM 없이 처리 가능한 것은 여기서 끝낸다.
   (완전중복 제거, 우선순위 정렬, 분량 예산 절단)
2. LLM 통합 - 모달리티가 2개 이상이거나 충돌 가능성이 있을 때만 호출한다.
   단일 모달리티 + 카드 소수면 LLM 호출을 아예 건너뛴다.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .clients import LLMError, StructuredLLMClient
from .schemas import (
    ConflictNote,
    DecoderTask,
    EvidenceCard,
    IntegratedEvidence,
    Level,
    ModalityEvidenceResult,
)

INTEGRATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kept_card_ids": {"type": "array", "items": {"type": "string"}},
        "dropped_card_ids": {"type": "array", "items": {"type": "string"}},
        "duplicate_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_ids": {"type": "array", "items": {"type": "string"}},
                    "representative": {"type": "string"},
                },
                "required": ["card_ids", "representative"],
                "additionalProperties": False,
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_ids": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "resolution": {"type": "string"},
                },
                "required": ["card_ids", "description", "resolution"],
                "additionalProperties": False,
            },
        },
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "covered": {"type": "boolean"},
                },
                "required": ["operation", "covered"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kept_card_ids", "dropped_card_ids", "duplicate_groups", "conflicts", "coverage"],
    "additionalProperties": False,
}

INTEGRATION_SYSTEM = """너는 멀티모달 RAG 시스템의 근거 통합 계층이다.

여러 모달리티(텍스트/이미지/영상) 디코더가 만든 근거 카드를 받아
최종 답변 디코더가 쓸 근거 집합으로 정리한다. 답변은 작성하지 않는다.

할 일
1. 중복: 같은 사실을 말하는 카드를 묶고 대표 1개만 남긴다. 모달리티가 다르면
   서로를 보강하는 것이므로 함부로 묶지 마라. 같은 사실일 때만 묶는다.
2. 충돌: 서로 모순되는 카드를 찾아 무엇이 충돌인지, 어느 쪽을 우선할지 적는다.
   검색점수와 confidence 가 높은 쪽을 우선하되 판단 근거를 resolution 에 남겨라.
3. 선별: 질문에 답하는 데 불필요한 카드는 dropped_card_ids 로 보낸다.
   근거가 희석되지 않게 남길 카드는 최대 {max_cards}개로 제한한다.
4. 커버리지: 각 required_operation 이 남은 카드로 수행 가능한지 판정한다.

kept_card_ids 와 dropped_card_ids 에는 실제로 주어진 card_id 만 쓴다."""


class EvidenceIntegrationLayer:
    def __init__(
        self,
        client: Optional[StructuredLLMClient] = None,
        max_cards: int = 12,
        char_budget: int = 4000,
        llm_min_cards: int = 4,
        duplicate_threshold: float = 0.85,
    ) -> None:
        self.client = client
        self.max_cards = max_cards
        self.char_budget = char_budget
        # 카드가 이보다 적고 모달리티가 하나면 LLM 통합을 건너뛴다.
        self.llm_min_cards = llm_min_cards
        self.duplicate_threshold = duplicate_threshold

    # ------------------------------------------------------------------

    def integrate(
        self,
        results: Sequence[ModalityEvidenceResult],
        context: DecoderTask,
    ) -> IntegratedEvidence:
        started = time.perf_counter()
        cards = [card for result in results for card in result.cards]

        if not cards:
            return IntegratedEvidence(
                coverage={op: False for op in context.required_operations},
                missing_operations=list(context.required_operations),
                char_budget=self.char_budget,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 1단계 - 규칙 기반 사전정리
        cards, duplicate_groups, dropped = self._prefilter(cards)

        modality_count = len({card.modality for card in cards})
        needs_llm = self.client is not None and (
            modality_count > 1
            or len(cards) >= self.llm_min_cards
            or context.complexity_level == Level.HIGH
        )

        if not needs_llm:
            kept, over_budget = self._apply_budget(cards)
            dropped.extend(over_budget)
            return IntegratedEvidence(
                cards=kept,
                dropped_card_ids=dropped,
                duplicate_groups=duplicate_groups,
                coverage=self._rule_coverage(kept, context),
                missing_operations=self._missing(kept, context),
                char_budget=self.char_budget,
                char_used=sum(card.char_cost() for card in kept),
                bypassed=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 2단계 - LLM 통합
        try:
            raw = self.client.generate_json(  # type: ignore[union-attr]
                INTEGRATION_SYSTEM.format(max_cards=self.max_cards),
                self._user_prompt(cards, results, context),
                INTEGRATION_SCHEMA,
            )
        except LLMError:
            # 통합에 실패해도 답변은 나와야 한다. 규칙 결과로 진행한다.
            kept, over_budget = self._apply_budget(cards)
            dropped.extend(over_budget)
            return IntegratedEvidence(
                cards=kept,
                dropped_card_ids=dropped,
                duplicate_groups=duplicate_groups,
                coverage=self._rule_coverage(kept, context),
                missing_operations=self._missing(kept, context),
                char_budget=self.char_budget,
                char_used=sum(card.char_cost() for card in kept),
                bypassed=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return self._apply_llm_result(
            raw, cards, duplicate_groups, dropped, context, started
        )

    # ------------------------------------------------------------------
    # 1단계
    # ------------------------------------------------------------------

    def _prefilter(self, cards: List[EvidenceCard]):
        """완전/근사 중복을 LLM 없이 제거하고 우선순위로 정렬한다."""
        cards = sorted(cards, key=lambda card: card.priority(), reverse=True)

        kept: List[EvidenceCard] = []
        groups: List[List[str]] = []
        dropped: List[str] = []

        for card in cards:
            duplicate_of = None
            for existing in kept:
                # 모달리티가 다르면 서로 보강하는 근거로 본다. 묶지 않는다.
                if existing.modality != card.modality:
                    continue
                if _similarity(existing.claim, card.claim) >= self.duplicate_threshold:
                    duplicate_of = existing
                    break

            if duplicate_of is None:
                kept.append(card)
                continue

            dropped.append(card.card_id)
            for group in groups:
                if duplicate_of.card_id in group:
                    group.append(card.card_id)
                    break
            else:
                groups.append([duplicate_of.card_id, card.card_id])

        return kept, groups, dropped

    def _apply_budget(self, cards: List[EvidenceCard]):
        """분량 예산. 근거 과부하로 답변 품질이 떨어지는 것을 막는다."""
        kept: List[EvidenceCard] = []
        dropped: List[str] = []
        used = 0
        for card in cards:
            cost = card.char_cost()
            if len(kept) >= self.max_cards or (used + cost > self.char_budget and kept):
                dropped.append(card.card_id)
                continue
            kept.append(card)
            used += cost
        return kept, dropped

    # ------------------------------------------------------------------
    # 2단계
    # ------------------------------------------------------------------

    def _user_prompt(
        self,
        cards: List[EvidenceCard],
        results: Sequence[ModalityEvidenceResult],
        context: DecoderTask,
    ) -> str:
        lines = [f"[원본 질문]\n{context.original_query}"]
        if context.required_operations:
            lines.append(f"[필요한 작업]\n{', '.join(context.required_operations)}")
        if context.constraints:
            lines.append(f"[제약]\n{', '.join(context.constraints)}")

        summaries = [
            f"- {result.modality.value}: {result.modality_summary}"
            for result in results
            if result.modality_summary
        ]
        if summaries:
            lines.append("[모달리티별 요약]\n" + "\n".join(summaries))

        insufficient = [
            f"- {result.modality.value}: {result.insufficient_reason}"
            for result in results
            if not result.is_sufficient and result.insufficient_reason
        ]
        if insufficient:
            lines.append("[근거 부족 신고]\n" + "\n".join(insufficient))

        card_lines = []
        for card in cards:
            card_lines.append(
                f"- card_id: {card.card_id} | 모달리티: {card.modality.value} | "
                f"관련도 {card.relevance:.2f} | 확신도 {card.confidence:.2f} | "
                f"검색점수 {card.retrieval_score:.4f}\n"
                f"  주장: {card.claim}\n"
                f"  상세: {card.detail}"
            )
        lines.append(f"[근거 카드 {len(cards)}개]\n" + "\n".join(card_lines))
        return "\n\n".join(lines)

    def _apply_llm_result(
        self,
        raw: Mapping[str, Any],
        cards: List[EvidenceCard],
        duplicate_groups: List[List[str]],
        dropped: List[str],
        context: DecoderTask,
        started: float,
    ) -> IntegratedEvidence:
        by_id = {card.card_id: card for card in cards}
        kept_ids = [cid for cid in (raw.get("kept_card_ids") or []) if cid in by_id]

        # LLM 이 전부 버리면 근거 없는 답변이 되므로 규칙 결과로 되돌린다.
        kept = [by_id[cid] for cid in kept_ids] if kept_ids else list(cards)
        kept.sort(key=lambda card: card.priority(), reverse=True)
        kept, over_budget = self._apply_budget(kept)

        dropped = list(dict.fromkeys(dropped + over_budget +
                       [cid for cid in (raw.get("dropped_card_ids") or []) if cid in by_id]))
        kept_id_set = {card.card_id for card in kept}
        dropped = [cid for cid in dropped if cid not in kept_id_set]

        for group in raw.get("duplicate_groups") or []:
            if isinstance(group, Mapping):
                ids = [cid for cid in (group.get("card_ids") or []) if cid in by_id]
                if len(ids) > 1:
                    duplicate_groups.append(ids)

        conflicts = [
            ConflictNote(
                card_ids=[cid for cid in (entry.get("card_ids") or []) if cid in by_id],
                description=str(entry.get("description", "") or ""),
                resolution=str(entry.get("resolution", "") or ""),
            )
            for entry in (raw.get("conflicts") or [])
            if isinstance(entry, Mapping) and entry.get("description")
        ]

        coverage: Dict[str, bool] = {}
        for entry in raw.get("coverage") or []:
            if isinstance(entry, Mapping) and entry.get("operation"):
                coverage[str(entry["operation"])] = bool(entry.get("covered"))
        for operation in context.required_operations:
            coverage.setdefault(operation, bool(kept))

        return IntegratedEvidence(
            cards=kept,
            dropped_card_ids=dropped,
            duplicate_groups=duplicate_groups,
            conflicts=conflicts,
            coverage=coverage,
            missing_operations=[op for op, ok in coverage.items() if not ok],
            char_budget=self.char_budget,
            char_used=sum(card.char_cost() for card in kept),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _rule_coverage(cards: List[EvidenceCard], context: DecoderTask) -> Dict[str, bool]:
        supported = {s for card in cards for s in card.supports}
        return {
            operation: (operation in supported) or bool(cards)
            for operation in context.required_operations
        }

    def _missing(self, cards: List[EvidenceCard], context: DecoderTask) -> List[str]:
        coverage = self._rule_coverage(cards, context)
        return [operation for operation, ok in coverage.items() if not ok]


_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def _similarity(left: str, right: str) -> float:
    """어절 기반 Jaccard. 형태소 분석기 의존성 없이 근사 중복만 잡는다."""
    a = set(_TOKEN_RE.findall(left.lower()))
    b = set(_TOKEN_RE.findall(right.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
