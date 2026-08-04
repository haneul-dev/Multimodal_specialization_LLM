# Evidence Decoder — 다층 디코더

`adaptive_rag` 가 반환한 표준 패킷을 받아 최종 답변까지 만드는 3계층 디코더.

```text
Adaptive RAG 패킷
    ↓ PacketAdapter
[1층] 모달리티별 근거 디코더   text / image / video   (병렬 실행)
    ↓ EvidenceCard[]
[2층] 근거 통합 계층           중복제거 · 충돌확인 · 우선순위 · 분량예산
    ↓ IntegratedEvidence
[3층] 최종 답변 디코더         + 원본 질문 + answer_constraints
```

## 설계 핵심

**모달리티와 무관한 단일 출력 규격.** 1층 디코더는 서로 다른 원본(텍스트/픽셀/영상)을 보지만
출력은 전부 `EvidenceCard` 다. 덕분에 2층은 모달리티를 몰라도 동작하고,
audio·table 이 추가돼도 2층 코드는 바뀌지 않는다.

**전체 패킷을 직접 받는다.** `adaptive_rag.build_decoder_inputs()` 는
`answer_constraints`, `complexity`, `uncertainty` 를 넘겨주지 않는다.
최종 답변 디코더는 `answer_constraints` 없이 답변 형식 제약을 지킬 수 없으므로,
`PacketAdapter` 가 `AdaptiveRAGOutput` 전체를 읽는다.
구형 `build_decoder_inputs()` 결과도 그대로 받아들이지만 그 경로에서는 답변 제약이 복구되지 않는다.

## 사용법

```python
from evidence_decoder import build_pipeline

pipeline = build_pipeline(asset_root="./data")
output = pipeline.run(rag_output)          # AdaptiveRAGOutput 또는 to_dict() 결과

print(output.final_answer.answer)
print(output.final_answer.citations)       # 근거로 쓴 card_id
print(output.trace.total_ms)               # 단계별 지연
```

`adaptive_rag` 와 이어 붙일 때:

```python
from adaptive_rag.adaptive_multimodal_rag_V3 import AdaptiveMultimodalRAGPipeline
from evidence_decoder import build_pipeline

rag_output = rag_pipeline.run(question, modality_summaries)
answer = build_pipeline(asset_root="./data").run(rag_output)
```

## 모델

| 계층 | 모델 | 선정 근거 |
|---|---|---|
| 텍스트 근거 디코더 | Upstage `solar-pro3` | 실측 1.30s로 최속이면서 카드 개수 지시를 정확히 준수 (`solar-mini` 5.89s·지시 위반, `solar-pro2` 1.75s) |
| 근거 통합 계층 | `solar-pro3` | 동일 |
| 최종 답변 디코더 | `solar-pro3` | 한국어 답변 품질 |
| 이미지·영상 디코더 | Google `gemini-3.5-flash` | 영상을 프레임 분해 없이 그대로 처리. 실측 5.09s로 후보 중 최속(3.6-flash 6.37s, flash-latest 6.80s) |

`solar-pro3` 는 `response_format: json_schema` strict 모드를 지원한다(실측 0.74s).
스키마가 서버에서 강제되므로 **JSON 파싱 실패로 인한 재시도가 구조적으로 사라진다.**

Upstage 3개 모델(`solar-mini`/`solar-pro2`/`solar-pro3`)은 모두 이미지 입력을 거부한다
(`Image input is not allowed for this model`). 그래서 비전은 별도 백엔드가 필요하다.

### 환경변수

```bash
UPSTAGE_API_KEY=up_...     # 필수 — 텍스트/통합/최종
GOOGLE_API_KEY=...         # 선택 — 이미지/영상 (Gemini Flash)
OPENAI_API_KEY=sk-...      # 선택 — Gemini 대신 사용
```

`gemini-2.5-flash` 는 신규 사용자에게 더 이상 제공되지 않는다(404). `-latest` 별칭은
모델이 바뀔 수 있어 실험 재현성을 위해 고정 이름을 쓴다.

**검증 완료 (2026-08-04)** — 실제 이미지(도표 399KB)와 영상(넙치 466KB)을 넣어
end-to-end 확인. 영상은 프레임 샘플링 없이 원본 그대로 전달했고, 캡션을 전혀 주지 않은
상태에서 "갈색 바탕에 흰색 반점이 있는 납작한 물고기들이 모래 바닥에 움직임 없이
멈춰 있다"고 정확히 판독했다. 지연은 이미지 6.2s / 영상 7.9s.

비전 키가 없으면 `CaptionFallbackVisionClient` 로 내려간다.
`evidence.metadata` 의 caption/OCR/자막 텍스트만으로 판단하고, 카드에 `degraded=True` 를 남겨
정상 경로와 실험에서 구분된다. `ResilientVisionClient` 가 런타임 401/429 도 같은 방식으로 흡수한다.

## 속도 설계

이 연구의 지표는 **RAG 도입에 따른 속도저하 억제**다. 세 가지 장치를 둔다.

1. **모달 디코더 병렬 실행** — 1층 지연이 모달리티 수만큼 누적되지 않고 최댓값으로 수렴한다.
   실측: text 3735ms + image 4853ms → 1층 4854ms (순차라면 8588ms).
2. **저복잡도 바이패스** — `complexity.level == low` 이고 근거가 적으면 1층을 건너뛰고
   검색 결과를 카드로 승격한다. LLM 호출 N회가 0회가 된다.
   원본 해석이 필수인 image/video/audio 는 바이패스 대상에서 제외된다.
3. **통합 계층 2단 구조** — 완전/근사 중복 제거, 우선순위 정렬, 분량 예산 절단은
   LLM 없이 규칙으로 끝낸다. 모달리티가 2개 이상이거나 카드가 많을 때만 LLM 을 호출한다.

## 실행

```bash
# 네트워크 없이 구조 검증
python -m evidence_decoder.test_offline

# 실제 LLM 1회 실행
python evidence_decoder/examples/run_live.py
python evidence_decoder/examples/run_live.py --packet rag_output.json --asset-root ./data

# 합성 패킷 생성 (앞단 없이 실험하기 위한 것)
python -m evidence_decoder.datagen --sweep latency --out packets_latency.json
python -m evidence_decoder.datagen --sweep dilution --out packets_dilution.json
python -m evidence_decoder.datagen --sweep integration --out packets_integration.json

# 지연/품질 비교 실험
python -m evidence_decoder.bench --packets packets_latency.json --arms raw full --group
```

### 합성 패킷을 쓰는 이유

앞단 `adaptive_rag` 의 코퍼스는 하드코딩 문장 3건이고 임베딩은 SHA256 해시 기반 난수라
(`example_usage_V3.py`), 검색 순위에 의미가 없다. 실제 데이터셋 교체 일정은 미정이다.

그리고 통합 계층이 하는 일(중복제거·충돌확인)은 애초에 실제 검색기로 측정할 수 없다.
검색 결과에는 "이 둘은 중복", "이 둘은 모순"이라는 정답 라벨이 붙지 않기 때문이다.
`datagen.py` 는 근거마다 역할(`_role`)을 심어 그 라벨을 만든다.

| 역할 | 측정 대상 |
|---|---|
| `gold` | 답변 정확도, 인용 정확도 |
| `irrelevant` | 근거 희석 억제율 |
| `duplicate` | 중복 제거율 |
| `contradictory` | 충돌 탐지 재현율 |
| `reinforcing` | 모달 간 중복 오탐 방지 |

공개 벤치마크(MultimodalQA 등)를 가공할 때도 같은 라벨 체계를 쓴다.

무관 근거는 기본이 **hard negative** 다. 주제가 아예 다른 잡음은 1층 디코더가
100% 걸러내 실험 변별력이 사라졌다(무관거부율 모든 구간 1.00). 실제 검색 잡음처럼
주제어를 공유하되 초점만 빗나가게 만든다 - 다른 대상·같은 측면, 같은 대상·다른 측면,
같은 대상·다른 매체. 쉬운 잡음 대조군은 `--sweep dilution-easy`.

### 품질 채점 (`scoring.py`)

```bash
python -m evidence_decoder.bench --packets packets_dilution.json \
       --arms raw no-integ full --group --score
```

LLM 심판을 최소로 쓴다. 1층·2층 지표는 전부 라벨 기반 규칙이라 재현 가능하고,
"LLM 이 만든 것을 LLM 이 채점하는" 순환을 피한다. 심판은 답변 요지 충족 여부에만 쓴다.

측정된 결과 (텍스트, 시나리오 3종 x 반복 2회 = 셀당 n=6, 2026-08-04)

**근거 희석 억제 - 1층의 기여**

| 무관 근거 비율 | raw 근거정밀 | 1층 있음 | raw 인용정밀 | 1층 있음 | raw 오염서술 | 1층 있음 |
|---|---|---|---|---|---|---|
| 0% | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| 40% | 0.60 | **1.00** | 0.78 | **1.00** | 0.67건 | **0건** |
| 57% | 0.43 | **1.00** | 0.70 | **1.00** | 0.33건 | **0건** |
| 66% | 0.33 | **0.92** | 0.51 | **0.92** | 0 | 0 |

시나리오별 일관성 (희석 전 구간 합산)

| 시나리오 | raw 근거정밀 | 1층 있음 |
|---|---|---|
| film | 0.59 | 0.94 |
| fish | 0.59 | 1.00 |
| battery | 0.59 | 1.00 |

세 주제에서 raw 근거정밀이 0.59 로 완전히 일치한다. 주제 의존성이 없다.

**중복·충돌 처리 - 2층의 기여**

| 패킷 유형 | 지표 | 통합 계층 없음 | 있음 |
|---|---|---|---|
| 충돌 | 충돌탐지 | 0.00 | **1.00** |
| 중복 | 중복제거 | 0.00 | 0.33 |
| 중복+충돌+잡음 | 중복제거 | 0.00 | **1.00** |
| 중복+충돌+잡음 | 충돌탐지 | 0.00 | **1.00** |
| 중복+충돌+잡음 | 인용정밀 | 0.60 | **0.72** |

시나리오별 일관성

| 시나리오 | 충돌탐지 | 중복제거 |
|---|---|---|
| film | 1.00 | 0.50 |
| fish | 1.00 | 1.00 |
| battery | 1.00 | **0.00** |

충돌 탐지는 주제와 무관하게 완전하지만, **중복 제거는 주제 편차가 크다.**
battery 시나리오에서는 전혀 잡지 못한다. 중복 문장이 원문에 없던 설명을
덧붙이면(공랭식 중복본이 "부품 수가 적고 중량 부담이 작았다"를 추가) 모델이
새로운 사실로 보는 것으로 추정된다. 미해결 과제다.

**답변 요지는 잡음에 강건하다.** 근거정밀이 0.33 까지 떨어져도 요지충족은
raw 에서도 1.00 을 유지한다. 희석의 피해는 요지 누락이 아니라 근거 오염
(근거정밀·인용정밀·오염서술)으로 나타난다. 세 시나리오 모두에서 동일하다.

**간결 모드가 품질을 해치는가** (희석 0%/66%, n=10)

| | gold채택 | 무관거부 | 근거정밀 | 인용정밀 | 요지충족 | 답변 길이 |
|---|---|---|---|---|---|---|
| 간결 off | 1.00 | 0.83 | 0.90 | 0.90 | 1.00 | 260자 |
| 간결 on | 1.00 | 0.87 | 0.93 | 0.93 | 1.00 | 222자 |

해치지 않는다. 답변이 15% 짧아졌지만 요지 충족은 동일하다. 군더더기만 줄었다.
단 텍스트 경로에서는 **지연 이득이 측정되지 않는다**(8376±1165 vs 9100±1537ms).
solar-pro3 는 TTFT 가 지배적이라 출력 토큰 감소가 지연에 반영되지 않는다.
간결 모드의 지연 이득은 비전 경로(-17%)에서만 유효하다.

## 결론의 신뢰도

이 실험들은 변동성이 크다. 지연의 표준편차가 중앙값의 30~40%에 달해
n=5~15 로는 20% 미만의 차이를 판정할 수 없다. 수치를 인용할 때 아래를 참고할 것.

| 결론 | 신뢰도 | 근거 |
|---|---|---|
| 근거 희석 억제는 1층의 기여 | 높음 | 시나리오 3종 전부에서 raw 0.59 vs 1층 0.94~1.00 |
| 충돌 탐지는 2층의 기여 | 높음 | 시나리오 3종 전부 0.00 vs 1.00 |
| 비전 경로 최적화 -61.8% | 높음 | 16895±2264 vs 6451±616ms, 범위 비중첩 |
| 답변 요지는 잡음에 강건 | 높음 | 근거정밀 0.33 에서도 요지충족 1.00, 3종 일치 |
| 간결 모드가 품질을 해치지 않음 | 중간 | n=10, 텍스트 경로만 확인 |
| 중복 제거 개선 | 낮음 | 시나리오 편차가 크다(0.00~1.00) |
| 간결 모드의 텍스트 지연 이득 | 없음 | 부호가 실행마다 뒤집힘 |
| 관련도 하한(min_relevance) 효과 | 없음 | 역효과, 원인 미규명 |

`bench` 비교군:

| 구성 | 1층 | 2층 | 용도 |
|---|---|---|---|
| `raw` | 없음 | 없음 | 베이스라인 — 검색 결과 직접 투입 |
| `no-integ` | 있음 | 없음 | 통합 계층 기여도 분리 |
| `bypass` | 조건부 | 있음 | 바이패스 효과 |
| `full` | 있음 | 있음 | 제안 구조 |

## 구성

| 파일 | 역할 |
|---|---|
| `schemas.py` | `EvidenceCard` 등 공통 스키마. 계층 간 계약이 전부 여기 있다 |
| `packet.py` | Adaptive RAG 패킷 → `DecoderTask` 어댑터 |
| `clients.py` | Solar / Gemini / OpenAI 클라이언트, 폴백 래퍼 (외부 SDK 의존 없음) |
| `assets.py` | 파일명 → 실제 이미지·영상 해석, 영상 프레임 샘플링 |
| `modality.py` | 1층 디코더 (텍스트 / 비전) |
| `integration.py` | 2층 근거 통합 계층 |
| `final_decoder.py` | 3층 최종 답변 디코더 |
| `pipeline.py` | 병렬 실행·바이패스·계측 |
| `datagen.py` | 합성 패킷 생성기 (근거 역할 라벨 포함) |
| `scoring.py` | 품질 채점기 (라벨 기반 규칙 + 답변 정확도 LLM 심판) |
| `bench.py` | 실험 하네스 |

## adaptive_rag 담당자에게 요청할 사항

1. `build_decoder_inputs()` 에 `answer_constraints` 와 `complexity.level` 추가.
   현재는 `PacketAdapter` 가 전체 패킷을 직접 읽어 우회하고 있다.
2. 이미지·영상 근거의 `metadata` 에 원본 경로(`path`)를 넣어줄 것.
   현재 `content` 는 `poster_003.jpg` 같은 파일명뿐이라 `asset_root` 하위를 재귀 탐색해야 한다.
3. 앞단 인코더가 만든 캡션·자막이 있다면 `metadata.caption` / `metadata.transcript` 로 전달.
   비전 백엔드가 죽었을 때의 폴백 품질이 여기서 결정된다.
