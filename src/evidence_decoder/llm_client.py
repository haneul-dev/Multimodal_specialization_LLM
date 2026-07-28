"""1층 디코더가 근거 추출에 사용하는 LLM 호출 추상화.

API 키가 없는 동안은 MockLLMClient로 파이프라인 배선을 검증하고,
GEMINI_API_KEY가 설정되면 자동으로 GeminiClient를 사용한다.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def extract(self, prompt: str) -> str:
        """prompt를 모델에 넣고 텍스트 응답을 반환한다."""


class MockLLMClient(LLMClient):
    """실제 API 없이 구조를 검증하기 위한 목 클라이언트.

    프롬프트에 담긴 원문(CONTENT: 이후)을 앞부분만 잘라 그대로 돌려줘
    추출형 디코더의 동작을 흉내낸다. 실제 LLM 대신 배선 검증용이다.
    """

    def extract(self, prompt: str) -> str:
        marker = "CONTENT:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip()[:200]
        return prompt.strip()[:200]


class GeminiClient(LLMClient):
    """Gemini API 실제 호출. API 키 준비 후 연결 예정."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY가 설정되지 않았습니다. .env에 키를 추가한 뒤 사용하세요."
            )
        self.model = model

    def extract(self, prompt: str) -> str:
        raise NotImplementedError(
            "google-genai 클라이언트 연결 예정 (API 키 발급 후 구현)"
        )


def get_default_client() -> LLMClient:
    """GEMINI_API_KEY가 있으면 실제 클라이언트를, 없으면 mock을 반환한다."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiClient()
    return MockLLMClient()
