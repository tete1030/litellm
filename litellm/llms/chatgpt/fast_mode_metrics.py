from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ChatGPTFastModeMetrics:
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
            "model_name",
            "model_id",
            "profile",
            "virtual_key",
            "requested_service_tier",
            "effective_service_tier",
        ]
        counter_definitions = {
            "requests": (
                "litellm_chatgpt_fast_mode_requests_total",
                "ChatGPT requests grouped by requested and effective service tier.",
            ),
            "prompt_tokens": (
                "litellm_chatgpt_fast_mode_prompt_tokens_total",
                "ChatGPT prompt tokens grouped by requested and effective service tier.",
            ),
            "completion_tokens": (
                "litellm_chatgpt_fast_mode_completion_tokens_total",
                "ChatGPT completion tokens grouped by requested and effective service tier.",
            ),
            "total_tokens": (
                "litellm_chatgpt_fast_mode_tokens_total",
                "ChatGPT total tokens grouped by requested and effective service tier.",
            ),
        }

        if registry is not None:
            return {
                name: Counter(
                    metric_name,
                    description,
                    labelnames=labelnames,
                    registry=registry,
                )
                for name, (metric_name, description) in counter_definitions.items()
            }

        with cls._default_lock:
            if cls._default_counters is None:
                cls._default_counters = {
                    name: Counter(metric_name, description, labelnames=labelnames)
                    for name, (metric_name, description) in counter_definitions.items()
                }
            return cls._default_counters

    @staticmethod
    def _normalize_service_tier_label(service_tier: Optional[str]) -> str:
        normalized = (service_tier or "").strip().lower()
        return normalized or "default"

    @staticmethod
    def _normalize_virtual_key_label(virtual_key: Optional[str]) -> str:
        normalized = str(virtual_key or "").strip()
        return normalized or "unknown"

    def _get_labels(
        self,
        *,
        model_name: str,
        model_id: str,
        profile: str,
        virtual_key: Optional[str],
        requested_service_tier: Optional[str],
        effective_service_tier: Optional[str],
    ) -> Dict[str, str]:
        return {
            "model_name": model_name,
            "model_id": model_id,
            "profile": profile,
            "virtual_key": self._normalize_virtual_key_label(virtual_key),
            "requested_service_tier": self._normalize_service_tier_label(
                requested_service_tier
            ),
            "effective_service_tier": self._normalize_service_tier_label(
                effective_service_tier
            ),
        }

    def observe_request(
        self,
        *,
        model_name: str,
        model_id: str,
        profile: str,
        requested_service_tier: Optional[str],
        effective_service_tier: Optional[str],
        virtual_key: Optional[str] = None,
    ) -> None:
        if not self._counters:
            return

        labels = self._get_labels(
            model_name=model_name,
            model_id=model_id,
            profile=profile,
            virtual_key=virtual_key,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        self._counters["requests"].labels(**labels).inc()

    def observe_usage(
        self,
        *,
        model_name: str,
        model_id: str,
        profile: str,
        requested_service_tier: Optional[str],
        effective_service_tier: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        virtual_key: Optional[str] = None,
    ) -> None:
        if not self._counters:
            return

        labels = self._get_labels(
            model_name=model_name,
            model_id=model_id,
            profile=profile,
            virtual_key=virtual_key,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        self._counters["prompt_tokens"].labels(**labels).inc(prompt_tokens)
        self._counters["completion_tokens"].labels(**labels).inc(completion_tokens)
        self._counters["total_tokens"].labels(**labels).inc(total_tokens)
