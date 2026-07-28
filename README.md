# 계층형 근거 해석 디코더 (연구 3)

Adaptive Evidence Processing 기반 멀티모달 RAG-LLM 파이프라인의 3단계 모듈.
검색된 멀티모달 근거를 LLM에 그대로 넣지 않고, 모달별로 핵심 근거를 추출한 뒤
통합해서 "짧지만 밀도 높은 근거"로 만들어 전달한다.

## 중요한 설계 결정

이 디코더는 **hard 여부와 무관하게 항상 실행**된다. 연구2가 넘기는 `hard_signal`은
실행 여부를 결정하지 않고, 내부 처리 깊이(`Depth.LIGHT` / `Depth.FULL`)만 조절한다.
(`src/evidence_decoder/pipeline.py`의 `resolve_depth` 참고)

## 구조

- `schema.py` — 근거 공통 스키마 (`RawEvidence`, `Claim`)
- `llm_client.py` — LLM 호출 추상화. API 키 없으면 `MockLLMClient` 자동 사용
- `text_decoder.py` — 1층 Text Evidence Decoder (구현 완료)
- `image_decoder.py`, `audio_decoder.py` — 1층 Image/Audio Evidence Decoder (Phase 3, 스텁)
- `integrated_decoder.py` — 2층 Integrated Evidence Decoder (중복 제거·정렬·토큰 예산)
- `pipeline.py` — `HierarchicalEvidenceDecoder` 전체 진입점

## 현재 상태

- Text Evidence Decoder + Integrated Evidence Decoder: mock LLM 기반으로 배선 검증 완료
- Image/Audio Evidence Decoder: 구조만 존재, 미구현 (근거가 있어도 스킵하고 로그만 남김)
- 실제 LLM: 아직 미연결. `GEMINI_API_KEY` 환경변수가 설정되면 자동으로 실제 클라이언트 사용

## 실행

```bash
python -m unittest discover -s tests
```

## 다음 단계

- Gemini API 키 발급 후 `llm_client.GeminiClient` 실제 구현 연결
- 연구1·2 실제 데이터 포맷 확정 후 mock 데이터를 교체
- Image/Audio Decoder 구현 (Phase 3)
- 충돌 탐지 로직 구현 (현재 `IntegratedEvidenceDecoder._find_conflicts`는 빈 구현)
