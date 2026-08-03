"""실험 하네스 - 속도저하 억제 효과 측정.

    python -m evidence_decoder.bench --repeat 3

비교군
  full        : 모달 디코더 + 통합 계층 전부 사용 (제안 구조)
  no-integ    : 통합 계층 제거 (카드 단순 이어붙이기)
  bypass      : 저복잡도 바이패스 허용
  raw         : 1층/2층 모두 없이 검색 결과를 최종 디코더에 직접 투입 (베이스라인)

측정: 총 지연, 단계별 지연, LLM 호출 수, 통합 전후 카드 수, 최종 답변 길이.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .clients import LLMError, build_default_clients
from .final_decoder import FinalAnswerDecoder
from .integration import EvidenceIntegrationLayer
from .modality import build_modality_decoders
from .packet import PacketAdapter
from .pipeline import MultiLayerDecoderPipeline, PipelineConfig
from .schemas import DecoderOutput, Level, Modality

ARMS = ("raw", "no-integ", "bypass", "full")


@dataclass
class ArmResult:
    arm: str
    total_ms: List[float] = field(default_factory=list)
    modality_ms: List[float] = field(default_factory=list)
    integration_ms: List[float] = field(default_factory=list)
    final_ms: List[float] = field(default_factory=list)
    llm_calls: List[int] = field(default_factory=list)
    cards_before: List[int] = field(default_factory=list)
    cards_after: List[int] = field(default_factory=list)
    answer_chars: List[int] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def record(self, output: DecoderOutput) -> None:
        trace = output.trace
        self.total_ms.append(trace.total_ms)
        self.modality_ms.append(trace.modality_stage_ms)
        self.integration_ms.append(trace.integration_ms)
        self.final_ms.append(trace.final_ms)
        self.llm_calls.append(trace.llm_calls)
        self.cards_before.append(trace.cards_before_integration)
        self.cards_after.append(trace.cards_after_integration)
        self.answer_chars.append(len(output.final_answer.answer))
        self.citations.append(len(output.final_answer.citations))

    def summary(self) -> Dict[str, Any]:
        def med(values: Sequence[float]) -> float:
            return round(statistics.median(values), 1) if values else 0.0

        return {
            "arm": self.arm,
            "n": len(self.total_ms),
            "total_ms_median": med(self.total_ms),
            "modality_ms_median": med(self.modality_ms),
            "integration_ms_median": med(self.integration_ms),
            "final_ms_median": med(self.final_ms),
            "llm_calls_median": med(self.llm_calls),
            "cards_before_median": med(self.cards_before),
            "cards_after_median": med(self.cards_after),
            "answer_chars_median": med(self.answer_chars),
            "citations_median": med(self.citations),
            "errors": len(self.errors),
        }


def build_arm(arm: str, clients: Dict[str, Any], asset_root: Optional[str]) -> MultiLayerDecoderPipeline:
    from .assets import AssetLoader

    text_client = clients["text"]
    loader = AssetLoader(asset_root=asset_root)
    decoders = build_modality_decoders(
        text_client, clients["vision"], loader,
        (Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
    )

    if arm == "raw":
        # 1층/2층 없음. 검색 결과가 그대로 최종 디코더로 간다.
        # 이미지/영상도 강제로 바이패스해야 진짜 "디코더 없음" 베이스라인이 된다.
        config = PipelineConfig(force_modality_bypass=True, enable_integration=False)
    elif arm == "no-integ":
        config = PipelineConfig(enable_modality_bypass=False, enable_integration=False)
    elif arm == "bypass":
        config = PipelineConfig(enable_modality_bypass=True, enable_integration=True)
    else:  # full
        config = PipelineConfig(enable_modality_bypass=False, enable_integration=True)

    return MultiLayerDecoderPipeline(
        modality_decoders=decoders,
        integration_layer=EvidenceIntegrationLayer(client=text_client),
        final_decoder=FinalAnswerDecoder(text_client),
        config=config,
    )


def run_bench(
    packets: List[Dict[str, Any]],
    arms: Sequence[str] = ARMS,
    repeat: int = 1,
    asset_root: Optional[str] = None,
) -> Dict[str, ArmResult]:
    clients = build_default_clients()
    print(f"비전 백엔드: {clients['vision_backend']}")

    # 워밍업. 첫 호출은 TCP/TLS 핸드셰이크와 모델 콜드스타트를 떠안으므로
    # 측정에 넣으면 첫 번째 구성만 불리해진다.
    if packets:
        try:
            build_arm("raw", clients, asset_root).run(packets[0])
            print("  (워밍업 1회 완료)")
        except Exception as error:  # noqa: BLE001
            print(f"  (워밍업 실패, 무시: {error})")

    results: Dict[str, ArmResult] = {arm: ArmResult(arm) for arm in arms}
    for arm in arms:
        pipeline = build_arm(arm, clients, asset_root)
        for index, packet in enumerate(packets):
            for _ in range(repeat):
                try:
                    output = pipeline.run(packet)
                    results[arm].record(output)
                    print(
                        f"  [{arm}] 패킷{index} "
                        f"{output.trace.total_ms:7.0f}ms  "
                        f"호출{output.trace.llm_calls}  "
                        f"카드{output.trace.cards_before_integration}->"
                        f"{output.trace.cards_after_integration}"
                    )
                except (LLMError, Exception) as error:  # noqa: BLE001
                    results[arm].errors.append(repr(error))
                    print(f"  [{arm}] 패킷{index} 실패: {error}")
    return results


def print_table(results: Dict[str, ArmResult]) -> None:
    rows = [result.summary() for result in results.values()]
    headers = [
        ("arm", "구성", 10),
        ("total_ms_median", "총지연ms", 10),
        ("modality_ms_median", "1층ms", 9),
        ("integration_ms_median", "2층ms", 9),
        ("final_ms_median", "최종ms", 9),
        ("llm_calls_median", "호출", 6),
        ("cards_before_median", "카드전", 7),
        ("cards_after_median", "카드후", 7),
        ("answer_chars_median", "답변자", 7),
        ("citations_median", "인용", 6),
        ("errors", "오류", 5),
    ]
    print("\n" + "=" * 100)
    print("".join(title.ljust(width) for _, title, width in headers))
    print("-" * 100)
    for row in rows:
        print("".join(str(row[key]).ljust(width) for key, _, width in headers))
    print("=" * 100)

    baseline = results.get("raw")
    full = results.get("full")
    if baseline and full and baseline.total_ms and full.total_ms:
        base = statistics.median(baseline.total_ms)
        proposed = statistics.median(full.total_ms)
        overhead = (proposed - base) / base * 100 if base else 0.0
        print(f"\n제안 구조의 지연 증가율: {overhead:+.1f}%  (raw {base:.0f}ms -> full {proposed:.0f}ms)")


def load_packets(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else [data]
    from .test_offline import FULL_PACKET

    return [FULL_PACKET]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="다층 디코더 지연/품질 비교")
    parser.add_argument("--packets", help="RAG 패킷 JSON 경로 (객체 또는 배열)")
    parser.add_argument("--asset-root", help="이미지/영상 원본 루트")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    args = parser.parse_args(argv)

    packets = load_packets(args.packets)
    print(f"패킷 {len(packets)}개 x 반복 {args.repeat}회 x 구성 {len(args.arms)}종")
    started = time.perf_counter()
    results = run_bench(packets, args.arms, args.repeat, args.asset_root)
    print_table(results)
    print(f"총 소요 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
