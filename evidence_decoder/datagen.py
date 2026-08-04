"""합성 패킷 생성기.

앞단 adaptive_rag 의 코퍼스와 인코더가 아직 더미라 실제 패킷을 받을 수 없다.
그래서 AdaptiveRAGOutput 과 동일한 형태(schema 3.0)의 패킷을 직접 만든다.

두 가지 용도를 겸한다.
1. 속도 실험 - 복잡도 / 모달조합 / 근거개수 축을 훑어 지연 구조를 본다.
   근거 내용의 의미는 중요하지 않고 분량과 개수만 맞으면 된다.
2. 품질 실험 - 근거마다 역할 라벨을 심는다. 공개 벤치마크를 가공할 때도
   같은 라벨 체계를 쓰므로, 그때는 _gold 필드만 실제 값으로 채우면 된다.

근거 역할 (EvidenceRole)
    gold          정답에 필요한 근거
    irrelevant    검색기가 흔히 섞어오는 무관 근거   -> 근거 희석 억제 측정
    duplicate     gold 와 같은 사실, 다른 표현       -> 중복 제거율 측정
    contradictory gold 와 반대되는 진술              -> 충돌 탐지 재현율 측정
    reinforcing   다른 모달리티가 같은 사실을 보강    -> 모달 간 오탐 방지 측정

실행
    python -m evidence_decoder.datagen --sweep latency --out packets_latency.json
    python -m evidence_decoder.datagen --sweep dilution --out packets_dilution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .schemas import Level, Modality


class EvidenceRole(str, Enum):
    GOLD = "gold"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"
    CONTRADICTORY = "contradictory"
    REINFORCING = "reinforcing"


# ============================================================
# 소재 - 실제 검색 결과와 비슷한 분량(200~400자)을 맞춘다.
# 지연은 토큰 수에 비례하므로 길이가 의미보다 중요하다.
# ============================================================

TOPIC = "영화 감독 A의 연출 스타일 변화"

GOLD_TEXTS = [
    "감독 A의 초기작은 인물 중심 서사와 느린 호흡을 특징으로 한다. 평론가들은 특히 장면 전환을 최소화하고 한 컷을 길게 유지하는 방식이 인물의 내면을 드러내는 데 기여했다고 평가했다. 이러한 연출은 데뷔작부터 세 번째 작품까지 일관되게 나타난다.",
    "최근작에서는 편집 속도가 눈에 띄게 빨라졌다. 평균 숏 길이가 이전 작품의 절반 수준으로 줄었으며, 교차 편집을 적극적으로 활용해 긴장감을 조성한다. 이는 장르적 관습을 의도적으로 끌어들인 결과로 해석된다.",
    "색채 설계에서도 변화가 확인된다. 초기작이 저채도의 청색 계열을 주조색으로 삼았다면, 최근작은 주황과 청록의 보색 대비를 전면에 내세운다. 이 변화는 촬영감독 교체 시점과 일치한다.",
    "인물 배치 방식이 대칭에서 비대칭으로 이동했다. 초기작에서 인물은 화면 중앙에 놓이는 경우가 많았으나, 최근작에서는 화면 가장자리로 밀려나거나 다른 피사체에 가려지는 구도가 반복된다.",
    "사운드 사용도 달라졌다. 초기작은 현장음과 침묵에 크게 의존했지만 최근작은 지속적인 배경 음악으로 정서를 유도한다. 음악 감독과의 협업 방식이 바뀐 것이 원인으로 지목된다.",
]

# GOLD_TEXTS 와 같은 순서. 정답 답변이 담아야 할 요지다.
# 답변 정확도 채점(scoring.py)의 기준이 된다.
GOLD_KEY_POINTS = [
    "초기작은 인물 중심 서사와 느린 호흡, 긴 컷을 사용했다",
    "최근작은 편집 속도가 빨라지고 평균 숏 길이가 짧아졌다",
    "색채가 저채도 청색에서 주황-청록 보색 대비로 바뀌었다",
    "인물 배치가 대칭에서 비대칭으로 이동했다",
    "사운드가 현장음·침묵 중심에서 지속적 배경 음악으로 바뀌었다",
]

# Hard negative. 주제가 아예 다른 잡음은 1층 디코더가 100% 걸러내서
# 실험에 변별력이 없었다(무관거부율 모든 구간 1.00). 실제 검색기가 섞어오는
# 잡음은 이런 모양이다 - 같은 주제어를 공유하면서 질문의 초점만 빗나간다.
#   (a) 대상은 맞고 측면이 다름
#   (b) 측면은 맞고 대상이 다름  <- 가장 위험. 답변에 다른 대상 정보가 섞인다
#   (c) 대상·측면 모두 맞지만 매체·시기가 다름
HARD_NEGATIVE_TEXTS = [
    # (b) 다른 감독, 같은 측면
    "감독 B의 최근작은 편집 속도를 오히려 늦추고 롱테이크 비중을 늘렸다. 평균 숏 길이가 이전 작품보다 길어졌으며 인물을 화면 중앙에 대칭으로 배치하는 구도가 두드러진다.",
    # (a) 같은 감독, 무관한 측면
    "감독 A는 대학에서 철학을 전공한 뒤 광고 대행사에서 5년간 근무했다. 영화계 입문은 단편 영화 공모전 수상을 계기로 이루어졌다.",
    # (c) 같은 감독·측면이지만 매체가 다름
    "감독 A가 연출한 자동차 광고는 3초 단위의 빠른 컷과 강한 보색 대비를 사용한다. 광고 특성상 짧은 시간에 시선을 붙잡아야 하기 때문이다.",
    # (b) 다른 감독, 같은 측면
    "감독 C의 색채 설계는 데뷔작부터 일관되게 주황과 청록의 보색 대비를 유지해 왔다. 촬영감독을 교체한 적이 없다는 점이 원인으로 지목된다.",
    # (a) 같은 감독, 무관한 측면
    "감독 A의 최근작은 제작비가 이전작의 세 배로 늘었고 해외 로케이션 촬영 비중이 커졌다. 배급사도 대형사로 바뀌었다.",
    # (c) 같은 감독·측면이지만 시기가 다름
    "감독 A가 학생 시절 만든 습작 단편들은 편집 실험이 두드러진다. 다만 본인은 이 작품들을 필모그래피에서 제외하고 있다.",
]

IRRELEVANT_TEXTS = [
    "1990년대 한국 영화 산업의 배급 구조는 대기업 자본의 유입으로 크게 재편되었다. 멀티플렉스 확산은 상영 편수와 스크린 점유 방식을 근본적으로 바꾸었으며, 이는 독립 영화의 상영 기회 축소로 이어졌다는 지적이 있다.",
    "영화제 수상 이력은 감독의 이후 제작비 규모와 상관관계를 보인다. 다만 이 상관이 인과인지에 대해서는 표본 편향 문제가 제기되어 왔으며, 수상 이전부터 투자 규모가 컸다는 반론도 존재한다.",
    "디지털 촬영으로의 전환은 후반 작업 비용 구조를 바꾸었다. 필름 현상 비용이 사라진 대신 색보정과 시각효과 비용이 증가했고, 전체 예산에서 후반 작업이 차지하는 비중이 높아졌다.",
    "관객 연령대별 선호 장르 조사에 따르면 20대는 액션과 스릴러를, 40대 이상은 드라마와 사극을 상대적으로 선호하는 경향이 나타났다. 이 조사는 표본 3천 명을 대상으로 진행되었다.",
    "영화 음악 저작권 계약은 통상 매절 방식과 러닝 개런티 방식으로 나뉜다. 최근에는 스트리밍 수익 배분 조항을 별도로 두는 사례가 늘고 있다.",
    "촬영 현장의 표준 근로 시간 규정이 도입되면서 회차당 촬영 분량 계획이 달라졌다. 제작 일정이 길어지는 대신 야간 촬영 비중은 줄어든 것으로 보고된다.",
    "포스터 디자인 외주 단가는 배급사 규모에 따라 편차가 크다. 대형 배급사는 전담 디자인 팀을 두는 경우가 많아 외주 비중이 상대적으로 낮다.",
    "영화 자막 번역 품질에 대한 논쟁은 오역 사례가 화제가 될 때마다 반복된다. 번역가의 작업 기간과 단가가 원인으로 지목되는 경우가 많다.",
]

# gold[0] 과 같은 사실, 다른 표현. 통합 계층이 묶어야 한다.
DUPLICATE_TEXTS = [
    "초기 작품들에서 감독 A는 긴 호흡의 촬영을 선호했다. 컷을 자주 나누지 않고 인물을 오래 담아내는 방식이 반복적으로 사용되었으며, 이는 등장인물의 심리를 관객에게 전달하는 장치로 기능했다.",
]

# gold[1] 과 반대되는 진술. 통합 계층이 충돌로 잡아야 한다.
CONTRADICTORY_TEXTS = [
    "최근작의 편집 리듬은 초기작과 크게 다르지 않다는 분석도 있다. 평균 숏 길이를 실제로 측정한 결과 유의미한 차이가 확인되지 않았으며, 빨라졌다는 인상은 음악과 카메라 움직임 때문이라는 것이다.",
]

IMAGE_CAPTIONS = {
    EvidenceRole.GOLD: [
        "저채도 청색 배경에 인물이 화면 중앙에 단독으로 배치된 초기작 포스터",
        "주황과 청록의 보색 대비가 강한 최근작 포스터, 인물은 화면 우측 하단에 치우쳐 있음",
        "명암 대비가 강한 흑백 스틸컷, 인물의 실루엣만 드러남",
    ],
    EvidenceRole.IRRELEVANT: [
        "영화제 레드카펫에서 촬영된 단체 사진",
        "제작 발표회 현장의 기자단 사진",
        "극장 로비에 설치된 입간판 사진",
    ],
    EvidenceRole.REINFORCING: [
        "초기작 포스터의 색상 분포 그래프, 청색 계열이 전체의 68%를 차지",
    ],
}

VIDEO_CAPTIONS = {
    EvidenceRole.GOLD: [
        "최근작 예고편 30초 구간, 12회의 컷 전환이 발생하며 평균 숏 길이 2.5초",
        "초기작 도입부 90초 롱테이크, 컷 전환 없이 인물을 추적",
    ],
    EvidenceRole.IRRELEVANT: [
        "감독 인터뷰 영상, 차기작 일정에 대한 질의응답",
    ],
    EvidenceRole.REINFORCING: [
        "최근작 색보정 비교 영상, 보정 전후의 색온도 차이를 나란히 제시",
    ],
}

QUESTIONS = {
    Level.LOW: (
        "감독 A의 최근작 편집 속도가 어떤지 알려줘.",
        ["describe"],
        ["3문장 이내로 답할 것"],
    ),
    Level.MEDIUM: (
        "감독 A의 초기작과 최근작의 연출 스타일 차이를 비교해줘.",
        ["compare", "describe"],
        ["차이를 항목별로 정리할 것"],
    ),
    Level.HIGH: (
        "감독 A의 연출 스타일이 어떻게 변했는지, 그 변화의 원인으로 지목되는 요인들과 함께 "
        "근거의 신뢰도까지 따져서 설명해줘.",
        ["compare", "describe", "verify", "analyze"],
        ["변화 항목별로 정리할 것", "근거가 엇갈리는 부분은 명시할 것"],
    ),
}


# ============================================================
# 생성
# ============================================================


@dataclass
class PacketSpec:
    """패킷 1개의 구성 조건."""

    group: str
    level: Level
    modalities: Sequence[Modality]
    gold_per_modality: int = 2
    irrelevant_per_modality: int = 0
    # True 면 주제를 공유하는 hard negative 를 쓴다. 쉬운 잡음은 1층이
    # 전부 걸러내 실험 변별력이 사라진다.
    hard_negatives: bool = True
    duplicates: int = 0
    contradictions: int = 0
    reinforcing: int = 0
    packet_id: str = ""


def _text_evidence(spec: PacketSpec) -> List[Dict[str, Any]]:
    entries: List[tuple] = []
    for i in range(spec.gold_per_modality):
        entries.append((EvidenceRole.GOLD, GOLD_TEXTS[i % len(GOLD_TEXTS)]))
    for i in range(spec.duplicates):
        entries.append((EvidenceRole.DUPLICATE, DUPLICATE_TEXTS[i % len(DUPLICATE_TEXTS)]))
    for i in range(spec.contradictions):
        entries.append((EvidenceRole.CONTRADICTORY, CONTRADICTORY_TEXTS[i % len(CONTRADICTORY_TEXTS)]))
    pool = HARD_NEGATIVE_TEXTS if spec.hard_negatives else IRRELEVANT_TEXTS
    for i in range(spec.irrelevant_per_modality):
        entries.append((EvidenceRole.IRRELEVANT, pool[i % len(pool)]))

    evidence = []
    for index, (role, content) in enumerate(entries):
        evidence.append(
            {
                "evidence_id": f"text_{index}",
                "modality": "text",
                # 검색 점수는 역할 순서대로 단조 감소시킨다. 실제 검색기가
                # 무관 근거를 하위에 두는 상황을 모사한다.
                "score": round(0.95 - 0.07 * index, 4),
                "content": content,
                "metadata": {"rank": index + 1, "_role": role.value},
            }
        )
    return evidence


def _media_evidence(
    spec: PacketSpec, modality: Modality, captions: Dict[EvidenceRole, List[str]]
) -> List[Dict[str, Any]]:
    entries: List[tuple] = []
    pool = captions.get(EvidenceRole.GOLD, [])
    for i in range(spec.gold_per_modality):
        entries.append((EvidenceRole.GOLD, pool[i % len(pool)]))
    for i in range(spec.reinforcing):
        pool_r = captions.get(EvidenceRole.REINFORCING) or pool
        entries.append((EvidenceRole.REINFORCING, pool_r[i % len(pool_r)]))
    pool_i = captions.get(EvidenceRole.IRRELEVANT, [])
    for i in range(spec.irrelevant_per_modality):
        if pool_i:
            entries.append((EvidenceRole.IRRELEVANT, pool_i[i % len(pool_i)]))

    ext = "jpg" if modality == Modality.IMAGE else "mp4"
    evidence = []
    for index, (role, caption) in enumerate(entries):
        evidence.append(
            {
                "evidence_id": f"{modality.value}_{index}",
                "modality": modality.value,
                "score": round(-0.03 - 0.05 * index, 4),
                "content": f"{modality.value}_{index:03d}.{ext}",
                # 원본 파일이 없으므로 caption 이 폴백 경로의 입력이 된다.
                # 실제 데이터로 바꿀 때는 metadata.path 를 추가하면 된다.
                "metadata": {"rank": index + 1, "caption": caption, "_role": role.value},
            }
        )
    return evidence


def build_packet(spec: PacketSpec) -> Dict[str, Any]:
    question, operations, answer_constraints = QUESTIONS[spec.level]

    retrieval_results: Dict[str, Any] = {}
    sub_queries: List[Dict[str, Any]] = []
    modality_focus: Dict[str, List[str]] = {}

    for modality in spec.modalities:
        if modality == Modality.TEXT:
            evidence = _text_evidence(spec)
            focus = ["연출 스타일", "편집", "서사"]
        elif modality == Modality.IMAGE:
            evidence = _media_evidence(spec, modality, IMAGE_CAPTIONS)
            focus = ["색조", "구도", "명암"]
        elif modality == Modality.VIDEO:
            evidence = _media_evidence(spec, modality, VIDEO_CAPTIONS)
            focus = ["편집 속도", "숏 길이", "카메라 움직임"]
        else:
            continue

        modality_focus[modality.value] = focus
        sub_queries.append(
            {"query": f"{TOPIC} {' '.join(focus[:2])}", "modality": modality.value, "priority": "high"}
        )
        retrieval_results[modality.value] = {
            "modality": modality.value,
            "query": f"{TOPIC} {' '.join(focus[:2])}",
            "candidate_k": 50,
            "final_k": len(evidence),
            "use_reranker": False,
            "uncertainty": {
                "top1_score": evidence[0]["score"] if evidence else 0.0,
                "top1_top2_gap": 0.05,
                "score_variance": 0.01,
                "shannon_entropy": 1.0,
                "normalized_entropy": 0.9,
                "level": "medium",
            },
            "evidence": evidence,
        }

    total = sum(len(r["evidence"]) for r in retrieval_results.values())
    return {
        "schema_version": "3.0",
        "original_query": question,
        "normalized_query": question,
        "query_context": {
            "input_context": "",
            "identified_entities": ["감독 A"],
            "required_operations": operations,
            "constraints": [],
            "modality_focus": modality_focus,
            "answer_constraints": answer_constraints,
            "retrieval_action": "retrieve",
            "sub_queries": sub_queries,
        },
        "complexity": {
            "level": spec.level.value,
            "score": {"low": 0.2, "medium": 0.55, "high": 0.85}[spec.level.value],
            "retrieval_demand": {m.value: spec.level.value for m in spec.modalities},
            "features": {},
            "reasons": [],
        },
        "retrieval_results": retrieval_results,
        # 파이프라인은 무시하고 실험 스크립트만 읽는 필드.
        "_meta": {
            "packet_id": spec.packet_id or spec.group,
            "group": spec.group,
            "level": spec.level.value,
            "modalities": [m.value for m in spec.modalities],
            "evidence_total": total,
            "gold_total": sum(
                1
                for r in retrieval_results.values()
                for e in r["evidence"]
                if e["metadata"]["_role"] == EvidenceRole.GOLD.value
            ),
            "irrelevant_total": sum(
                1
                for r in retrieval_results.values()
                for e in r["evidence"]
                if e["metadata"]["_role"] == EvidenceRole.IRRELEVANT.value
            ),
            "duplicate_total": spec.duplicates,
            "contradiction_total": spec.contradictions,
            # 채점 기준 (scoring.py 가 읽는다)
            "key_points": GOLD_KEY_POINTS[: spec.gold_per_modality]
            if Modality.TEXT in spec.modalities
            else [],
            "irrelevant_topics": [
                t.split(".")[0][:60]
                for t in (HARD_NEGATIVE_TEXTS if spec.hard_negatives else IRRELEVANT_TEXTS)[
                    : spec.irrelevant_per_modality
                ]
            ],
            "hard_negatives": spec.hard_negatives,
            "has_conflict": spec.contradictions > 0,
        },
    }


# ============================================================
# 스윕
# ============================================================


def sweep_latency() -> List[Dict[str, Any]]:
    """지연 구조 파악용. 복잡도 x 모달조합 x 근거개수."""
    combos = [
        ("T", [Modality.TEXT]),
        ("TI", [Modality.TEXT, Modality.IMAGE]),
        ("TIV", [Modality.TEXT, Modality.IMAGE, Modality.VIDEO]),
    ]
    specs: List[PacketSpec] = []
    for level in (Level.LOW, Level.MEDIUM, Level.HIGH):
        for tag, modalities in combos:
            for gold in (1, 3):
                specs.append(
                    PacketSpec(
                        group=f"{level.value}/{tag}/g{gold}",
                        level=level,
                        modalities=modalities,
                        gold_per_modality=gold,
                        packet_id=f"lat_{level.value}_{tag}_g{gold}",
                    )
                )
    return [build_packet(spec) for spec in specs]


def sweep_dilution() -> List[Dict[str, Any]]:
    """근거 희석 실험용. 무관 근거 비율을 0 -> 높음으로 올린다.

    통합 계층이 있는 구성과 없는 구성의 답변 품질 격차가 벌어져야 한다.

    텍스트 전용이다. 희석은 근거 개수와 잡음 비율의 문제라 모달리티와 무관하고,
    비전 백엔드 할당량에 실험이 묶이지 않게 하기 위함이다.
    멀티모달 희석은 sweep_dilution_multimodal 로 따로 잰다.
    """
    specs: List[PacketSpec] = []
    for irrelevant in (0, 2, 4, 6):
        ratio = irrelevant / (3 + irrelevant)
        specs.append(
            PacketSpec(
                group=f"dilution/{int(ratio * 100)}%",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT],
                gold_per_modality=3,
                irrelevant_per_modality=irrelevant,
                packet_id=f"dil_{irrelevant}",
            )
        )
    return [build_packet(spec) for spec in specs]


def sweep_dilution_multimodal() -> List[Dict[str, Any]]:
    """희석 실험의 멀티모달 판. 비전 백엔드가 살아 있을 때 쓴다."""
    specs = [
        PacketSpec(
            group=f"dilution-mm/{int(irrelevant / (3 + irrelevant) * 100)}%",
            level=Level.MEDIUM,
            modalities=[Modality.TEXT, Modality.IMAGE],
            gold_per_modality=3,
            irrelevant_per_modality=irrelevant,
            packet_id=f"dilmm_{irrelevant}",
        )
        for irrelevant in (0, 2, 4, 6)
    ]
    return [build_packet(spec) for spec in specs]


def sweep_integration() -> List[Dict[str, Any]]:
    """통합 계층 4대 기능 검증용. 중복/충돌/보강을 명시적으로 심는다."""
    return [
        build_packet(
            PacketSpec(
                group="integ/duplicate",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT],
                gold_per_modality=3,
                duplicates=1,
                packet_id="integ_dup",
            )
        ),
        build_packet(
            PacketSpec(
                group="integ/conflict",
                level=Level.HIGH,
                modalities=[Modality.TEXT],
                gold_per_modality=3,
                contradictions=1,
                packet_id="integ_conf",
            )
        ),
        build_packet(
            PacketSpec(
                group="integ/reinforce",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT, Modality.IMAGE],
                gold_per_modality=2,
                reinforcing=1,
                packet_id="integ_reinf",
            )
        ),
        build_packet(
            PacketSpec(
                group="integ/all",
                level=Level.HIGH,
                modalities=[Modality.TEXT, Modality.IMAGE, Modality.VIDEO],
                gold_per_modality=3,
                irrelevant_per_modality=2,
                duplicates=1,
                contradictions=1,
                reinforcing=1,
                packet_id="integ_all",
            )
        ),
    ]


def sweep_dilution_easy() -> List[Dict[str, Any]]:
    """주제가 아예 다른 쉬운 잡음. hard negative 와의 대조군."""
    return [
        build_packet(
            PacketSpec(
                group=f"dilution-easy/{int(n / (3 + n) * 100)}%",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT],
                gold_per_modality=3,
                irrelevant_per_modality=n,
                hard_negatives=False,
                packet_id=f"dileasy_{n}",
            )
        )
        for n in (0, 2, 4, 6)
    ]


SWEEPS = {
    "latency": sweep_latency,
    "dilution-easy": sweep_dilution_easy,
    "dilution": sweep_dilution,
    "dilution-mm": sweep_dilution_multimodal,
    "integration": sweep_integration,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="합성 RAG 패킷 생성")
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="latency")
    parser.add_argument("--out", help="저장 경로 (없으면 stdout)")
    args = parser.parse_args(argv)

    packets = SWEEPS[args.sweep]()
    payload = json.dumps(packets, ensure_ascii=False, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"{args.sweep}: 패킷 {len(packets)}개 -> {args.out}")
        for packet in packets:
            meta = packet["_meta"]
            print(
                f"  {meta['packet_id']:24s} {meta['group']:20s} "
                f"근거 {meta['evidence_total']:2d} "
                f"(gold {meta['gold_total']}, 무관 {meta['irrelevant_total']}, "
                f"중복 {meta['duplicate_total']}, 충돌 {meta['contradiction_total']})"
            )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
