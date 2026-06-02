from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Literal, Optional

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.llms.chatgpt.usage_service import (
    ChatGPTUsageService,
    UsageResult,
    _effective_available_from_result,
    is_usage_result_expired,
)
from litellm.types.llms.openai import AllMessageValues


@dataclass(frozen=True)
class ChatGPTUsageRoutingDecision:
    action: Literal["allow", "degrade", "block"]
    reason: Optional[str] = None
    blocked_window_labels: tuple[str, ...] = ()
    suggested_weight_multiplier: Optional[float] = None


class ChatGPTUsageHealthCheck(CustomLogger):
    def __init__(
        self,
        usage_service: ChatGPTUsageService,
        *,
        block_at_used_percent: float = 100.0,
    ) -> None:
        self.usage_service = usage_service
        self.block_at_used_percent = block_at_used_percent

    @staticmethod
    def _is_blocking_usage_error(error: Optional[str]) -> bool:
        return ChatGPTUsageHealthCheck._blocking_error_reason(error) is not None

    @staticmethod
    def _blocking_error_reason(error: Optional[str]) -> Optional[str]:
        if not error:
            return None

        normalized_error = error.lower()
        if (
            "usage auth failed" in normalized_error
            or "usage request failed (401)" in normalized_error
        ):
            return "usage_auth_unavailable"

        if (
            "usage request failed (402)" in normalized_error
            or "deactivated_workspace" in normalized_error
        ):
            return "usage_endpoint_unavailable"

        return None

    @staticmethod
    def evaluate_usage_result(
        usage_result: UsageResult,
        *,
        block_at_used_percent: float = 100.0,
        now: Optional[float] = None,
    ) -> ChatGPTUsageRoutingDecision:
        reference_time = now if now is not None else time.time()
        if usage_result.status != "ok":
            reason = ChatGPTUsageHealthCheck._blocking_error_reason(usage_result.error)
            if reason is not None:
                return ChatGPTUsageRoutingDecision(
                    action="block",
                    reason=reason,
                )
            return ChatGPTUsageRoutingDecision(action="allow")

        if is_usage_result_expired(usage_result, now=reference_time):
            return ChatGPTUsageRoutingDecision(
                action="block",
                reason="subscription_expired",
            )

        if not _effective_available_from_result(usage_result):
            return ChatGPTUsageRoutingDecision(
                action="block",
                reason="usage_profile_unavailable",
            )

        blocked_windows = tuple(
            window.label
            for window in usage_result.windows
            if window.used_percent >= block_at_used_percent
        )
        if blocked_windows:
            return ChatGPTUsageRoutingDecision(
                action="block",
                reason="usage_window_exhausted",
                blocked_window_labels=blocked_windows,
            )

        return ChatGPTUsageRoutingDecision(action="allow")

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        if isinstance(healthy_deployments, dict):
            healthy_deployments = [healthy_deployments]

        if len(healthy_deployments) == 0:
            return healthy_deployments

        self.usage_service.register_deployments(healthy_deployments)
        self.usage_service.ensure_background_refresh_task()

        allowed_deployments: List[dict] = []
        blocked_deployments: List[dict] = []

        for deployment in healthy_deployments:
            profile = self.usage_service.get_profile_name_from_deployment(deployment)
            if not profile:
                allowed_deployments.append(deployment)
                continue

            snapshot = await self.usage_service.get_snapshot(profile)
            if snapshot is None:
                allowed_deployments.append(deployment)
                continue

            decision = self.evaluate_usage_result(
                snapshot.result,
                block_at_used_percent=self.block_at_used_percent,
                now=time.time(),
            )
            if decision.action == "block":
                blocked_deployments.append(deployment)
                verbose_router_logger.debug(
                    "ChatGPTUsageHealthCheck: blocking deployment=%s profile=%s model=%s reason=%s exhausted_windows=%s",
                    deployment.get("model_info", {}).get("id"),
                    profile,
                    model,
                    decision.reason,
                    ",".join(decision.blocked_window_labels),
                )
                continue

            allowed_deployments.append(deployment)

        if allowed_deployments:
            return allowed_deployments

        if blocked_deployments:
            verbose_router_logger.debug(
                "ChatGPTUsageHealthCheck: all healthy deployments filtered for model=%s due to usage health check",
                model,
            )
        return []
