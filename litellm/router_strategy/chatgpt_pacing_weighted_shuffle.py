import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_router_logger
from litellm.llms.chatgpt.usage_service import (
    ChatGPTPacingInfo,
    ChatGPTUsageService,
    compute_chatgpt_pacing_info,
)
from litellm.router_utils.pre_call_checks.chatgpt_usage_health_check import (
    ChatGPTUsageHealthCheck,
)
from litellm.types.utils import LiteLLMPydanticObjectBase


class RoutingArgs(LiteLLMPydanticObjectBase):
    base_weight_exponent: float = 1.0
    underuse_boost_exponent: float = 1.2
    overuse_penalty_exponent: float = 2.0
    min_time_ratio: float = 0.02


class ChatGPTRoutingWeightMetrics:
    _default_lock = threading.Lock()
    _default_gauges: Optional[Dict[str, Any]] = None

    def __init__(self, registry: Any = None):
        self._gauges = self._build_gauges(registry)

    @classmethod
    def _build_gauges(cls, registry: Any = None) -> Dict[str, Any]:
        try:
            from prometheus_client import Gauge
        except ModuleNotFoundError:
            return {}

        labelnames = ["model_name", "model_id", "profile"]
        gauge_definitions = {
            "base_weight": (
                "litellm_chatgpt_routing_base_weight",
                "Static base weight configured for the deployment.",
            ),
            "weekly_remaining_ratio": (
                "litellm_chatgpt_routing_weekly_remaining_ratio",
                "Remaining ratio for the weekly pacing window.",
            ),
            "time_ratio": (
                "litellm_chatgpt_routing_time_ratio",
                "Normalized remaining time ratio until the earliest routing deadline.",
            ),
            "pace_ratio": (
                "litellm_chatgpt_routing_pace_ratio",
                "Ratio between remaining weekly quota and remaining time.",
            ),
            "pacing_factor": (
                "litellm_chatgpt_routing_pacing_factor",
                "Dynamic pacing multiplier applied on top of the base weight.",
            ),
            "effective_weight": (
                "litellm_chatgpt_routing_effective_weight",
                "Final effective weight used for deployment selection.",
            ),
            "effective_deadline": (
                "litellm_chatgpt_routing_effective_deadline_timestamp_seconds",
                "Earliest routing deadline from weekly reset and subscription expiry.",
            ),
        }

        if registry is not None:
            return {
                name: Gauge(metric_name, description, labelnames=labelnames, registry=registry)
                for name, (metric_name, description) in gauge_definitions.items()
            }

        with cls._default_lock:
            if cls._default_gauges is None:
                cls._default_gauges = {
                    name: Gauge(metric_name, description, labelnames=labelnames)
                    for name, (metric_name, description) in gauge_definitions.items()
                }
            return cls._default_gauges

    def observe(
        self,
        *,
        model_name: str,
        model_id: str,
        profile: str,
        base_weight: float,
        effective_weight: float,
        pacing_factor: Optional[float],
        pacing_info: Optional[ChatGPTPacingInfo],
    ) -> None:
        if not self._gauges:
            return

        labels = {
            "model_name": model_name,
            "model_id": model_id,
            "profile": profile,
        }
        self._gauges["base_weight"].labels(**labels).set(base_weight)
        self._gauges["effective_weight"].labels(**labels).set(effective_weight)
        self._gauges["weekly_remaining_ratio"].labels(**labels).set(
            pacing_info.remaining_ratio if pacing_info is not None else float("nan")
        )
        self._gauges["time_ratio"].labels(**labels).set(
            pacing_info.time_ratio
            if pacing_info is not None and pacing_info.time_ratio is not None
            else float("nan")
        )
        self._gauges["pace_ratio"].labels(**labels).set(
            pacing_info.pace_ratio
            if pacing_info is not None and pacing_info.pace_ratio is not None
            else float("nan")
        )
        self._gauges["pacing_factor"].labels(**labels).set(
            pacing_factor if pacing_factor is not None else float("nan")
        )
        self._gauges["effective_deadline"].labels(**labels).set(
            float(pacing_info.effective_deadline_at)
            if pacing_info is not None
            and pacing_info.effective_deadline_at is not None
            else float("nan")
        )


class ChatGPTPacingWeightedShuffle:
    def __init__(
        self,
        usage_service: ChatGPTUsageService,
        routing_args: Optional[dict] = None,
        metrics: Optional[ChatGPTRoutingWeightMetrics] = None,
    ):
        self.usage_service = usage_service
        self.routing_args = RoutingArgs(**(routing_args or {}))
        self.metrics = metrics or ChatGPTRoutingWeightMetrics()

    def _get_base_weight(self, deployment: Dict[str, Any]) -> float:
        candidates = [
            deployment.get("litellm_params", {}).get("weight"),
            deployment.get("weight"),
            deployment.get("model_info", {}).get("weight"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
        return 1.0

    def _get_effective_weight_for_snapshot(
        self,
        *,
        base_weight: float,
        snapshot: Any,
        reference_time: float,
    ) -> Tuple[float, Optional[float], Optional[ChatGPTPacingInfo], Optional[str]]:
        effective_weight = base_weight ** self.routing_args.base_weight_exponent
        if snapshot is None:
            return effective_weight, None, None, None

        decision = ChatGPTUsageHealthCheck.evaluate_usage_result(
            snapshot.result,
            now=reference_time,
        )
        if decision.action == "block":
            return 0.0, None, None, decision.reason

        pacing_info = compute_chatgpt_pacing_info(
            snapshot.result,
            now=reference_time,
            min_time_ratio=self.routing_args.min_time_ratio,
        )
        if pacing_info is None or pacing_info.time_ratio is None or pacing_info.pace_ratio is None:
            return effective_weight, None, pacing_info, None

        if pacing_info.pace_ratio >= 1.0:
            pacing_factor = pacing_info.pace_ratio ** self.routing_args.underuse_boost_exponent
        else:
            pacing_factor = pacing_info.pace_ratio ** self.routing_args.overuse_penalty_exponent
        return effective_weight * pacing_factor, pacing_factor, pacing_info, None

    def _choose_weighted_deployment(
        self,
        *,
        model_group: str,
        candidates_with_weights: List[Tuple[Dict[str, Any], float]],
    ) -> Optional[Dict[str, Any]]:
        weighted_candidates = [
            (deployment, weight)
            for deployment, weight in candidates_with_weights
            if weight > 0
        ]
        if not weighted_candidates:
            return None

        weights = [weight for _, weight in weighted_candidates]
        selected_index = random.choices(range(len(weights)), weights=weights)[0]
        selected_deployment = weighted_candidates[selected_index][0]
        verbose_router_logger.info(
            "chatgpt-pacing-weighted-shuffle selected deployment=%s model_group=%s weights=%s",
            selected_deployment.get("model_info", {}).get("id"),
            model_group,
            weights,
        )
        return selected_deployment

    def _observe_metrics(
        self,
        *,
        deployment: Dict[str, Any],
        profile: str,
        base_weight: float,
        effective_weight: float,
        pacing_factor: Optional[float],
        pacing_info: Optional[ChatGPTPacingInfo],
    ) -> None:
        self.metrics.observe(
            model_name=str(deployment.get("model_name", "") or ""),
            model_id=str(deployment.get("model_info", {}).get("id", "") or ""),
            profile=profile,
            base_weight=base_weight,
            effective_weight=effective_weight,
            pacing_factor=pacing_factor,
            pacing_info=pacing_info,
        )

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Any] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        del messages, input, request_kwargs
        self.usage_service.register_deployments(healthy_deployments)
        self.usage_service.ensure_background_refresh_task()
        reference_time = time.time()

        candidates_with_weights: List[Tuple[Dict[str, Any], float]] = []
        for deployment in healthy_deployments:
            profile = self.usage_service.get_profile_name_from_deployment(deployment)
            base_weight = self._get_base_weight(deployment)
            if not profile:
                candidates_with_weights.append(
                    (deployment, base_weight ** self.routing_args.base_weight_exponent)
                )
                continue

            snapshot = await self.usage_service.get_snapshot(profile)
            effective_weight, pacing_factor, pacing_info, blocked_reason = (
                self._get_effective_weight_for_snapshot(
                    base_weight=base_weight,
                    snapshot=snapshot,
                    reference_time=reference_time,
                )
            )
            self._observe_metrics(
                deployment=deployment,
                profile=profile,
                base_weight=base_weight,
                effective_weight=effective_weight,
                pacing_factor=pacing_factor,
                pacing_info=pacing_info,
            )
            if blocked_reason is not None:
                verbose_router_logger.debug(
                    "chatgpt-pacing-weighted-shuffle blocking deployment=%s profile=%s model_group=%s reason=%s",
                    deployment.get("model_info", {}).get("id"),
                    profile,
                    model_group,
                    blocked_reason,
                )
            candidates_with_weights.append((deployment, effective_weight))

        return self._choose_weighted_deployment(
            model_group=model_group,
            candidates_with_weights=candidates_with_weights,
        )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Any] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        del messages, input, request_kwargs
        self.usage_service.register_deployments(healthy_deployments)
        reference_time = time.time()

        candidates_with_weights: List[Tuple[Dict[str, Any], float]] = []
        for deployment in healthy_deployments:
            profile = self.usage_service.get_profile_name_from_deployment(deployment)
            base_weight = self._get_base_weight(deployment)
            if not profile:
                candidates_with_weights.append(
                    (deployment, base_weight ** self.routing_args.base_weight_exponent)
                )
                continue

            snapshot = self.usage_service.get_snapshot_sync(profile)
            effective_weight, pacing_factor, pacing_info, blocked_reason = (
                self._get_effective_weight_for_snapshot(
                    base_weight=base_weight,
                    snapshot=snapshot,
                    reference_time=reference_time,
                )
            )
            self._observe_metrics(
                deployment=deployment,
                profile=profile,
                base_weight=base_weight,
                effective_weight=effective_weight,
                pacing_factor=pacing_factor,
                pacing_info=pacing_info,
            )
            if blocked_reason is not None:
                verbose_router_logger.debug(
                    "chatgpt-pacing-weighted-shuffle blocking deployment=%s profile=%s model_group=%s reason=%s",
                    deployment.get("model_info", {}).get("id"),
                    profile,
                    model_group,
                    blocked_reason,
                )
            candidates_with_weights.append((deployment, effective_weight))

        return self._choose_weighted_deployment(
            model_group=model_group,
            candidates_with_weights=candidates_with_weights,
        )
