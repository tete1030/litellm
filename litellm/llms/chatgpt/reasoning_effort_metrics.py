from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ChatGPTReasoningEffortMetrics:
    _default_lock = threading.Lock()
    _default_counters: Optional[Dict[str, Any]] = None

    def __init__(self, registry: Any = None):
        self._counters = self._build_counters(registry)

    @classmethod
    def _build_counters(cls, registry: Any = None) -> Dict[str, Any]:
        try:
            from prometheus_client import Counter
        except ModuleNotFoundError:
            return {}

        labelnames = [
            "model",
            "virtual_key",
            "requested_effort",
            "effective_effort",
            "action",
        ]
        definitions = {
            "decisions": (
                "litellm_chatgpt_reasoning_effort_decisions_total",
                "ChatGPT reasoning effort policy decisions.",
            ),
            "prompt_tokens": (
                "litellm_chatgpt_reasoning_effort_prompt_tokens_total",
                "ChatGPT prompt tokens grouped by reasoning effort decision.",
            ),
            "completion_tokens": (
                "litellm_chatgpt_reasoning_effort_completion_tokens_total",
                "ChatGPT completion tokens grouped by reasoning effort decision.",
            ),
            "reasoning_tokens": (
                "litellm_chatgpt_reasoning_effort_reasoning_tokens_total",
                "ChatGPT reported reasoning tokens grouped by reasoning effort decision.",
            ),
            "total_tokens": (
                "litellm_chatgpt_reasoning_effort_tokens_total",
                "ChatGPT total tokens grouped by reasoning effort decision.",
            ),
        }

        if registry is not None:
            return {
                key: Counter(name, description, labelnames=labelnames, registry=registry)
                for key, (name, description) in definitions.items()
            }

        with cls._default_lock:
            if cls._default_counters is None:
                cls._default_counters = {
                    key: Counter(name, description, labelnames=labelnames)
                    for key, (name, description) in definitions.items()
                }
            return cls._default_counters

    @staticmethod
    def _normalize_label(value: Optional[str], default: str) -> str:
        normalized = str(value or "").strip().lower()
        return normalized or default

    def _labels(
        self,
        *,
        model: str,
        virtual_key: Optional[str],
        requested_effort: Optional[str],
        effective_effort: Optional[str],
        action: Optional[str],
    ) -> Dict[str, str]:
        return {
            "model": self._normalize_label(model, "unknown"),
            "virtual_key": str(virtual_key or "").strip() or "unknown",
            "requested_effort": self._normalize_label(
                requested_effort, "default"
            ),
            "effective_effort": self._normalize_label(
                effective_effort, "default"
            ),
            "action": self._normalize_label(action, "allow"),
        }

    def observe_decision(
        self,
        *,
        model: str,
        virtual_key: Optional[str],
        requested_effort: Optional[str],
        effective_effort: Optional[str],
        action: Optional[str],
    ) -> None:
        if not self._counters:
            return
        labels = self._labels(
            model=model,
            virtual_key=virtual_key,
            requested_effort=requested_effort,
            effective_effort=effective_effort,
            action=action,
        )
        self._counters["decisions"].labels(**labels).inc()

    def observe_usage(
        self,
        *,
        model: str,
        virtual_key: Optional[str],
        requested_effort: Optional[str],
        effective_effort: Optional[str],
        action: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: Optional[int] = None,
    ) -> None:
        if not self._counters:
            return
        labels = self._labels(
            model=model,
            virtual_key=virtual_key,
            requested_effort=requested_effort,
            effective_effort=effective_effort,
            action=action,
        )
        self._counters["prompt_tokens"].labels(**labels).inc(prompt_tokens)
        self._counters["completion_tokens"].labels(**labels).inc(completion_tokens)
        self._counters["total_tokens"].labels(**labels).inc(total_tokens)
        if reasoning_tokens is not None:
            self._counters["reasoning_tokens"].labels(**labels).inc(
                reasoning_tokens
            )


_default_metrics: Optional[ChatGPTReasoningEffortMetrics] = None
_default_metrics_lock = threading.Lock()


def get_chatgpt_reasoning_effort_metrics() -> ChatGPTReasoningEffortMetrics:
    global _default_metrics
    with _default_metrics_lock:
        if _default_metrics is None:
            _default_metrics = ChatGPTReasoningEffortMetrics()
        return _default_metrics
