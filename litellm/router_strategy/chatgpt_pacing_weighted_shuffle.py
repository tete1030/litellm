import math
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_FAST_SERVICE_TIER,
    chatgpt_fast_mode_allowance,
    chatgpt_fast_mode_allowed,
    normalize_chatgpt_service_tier,
)
from litellm.llms.chatgpt.fast_mode_metrics import ChatGPTFastModeMetrics
from litellm.llms.chatgpt.usage_service import (
    ChatGPTPacingInfo,
    ChatGPTUsageService,
    build_codex_rate_limit_error,
    compute_chatgpt_pacing_info,
)
from litellm.router_utils.pre_call_checks.chatgpt_usage_health_check import (
    ChatGPTUsageHealthCheck,
)
from litellm.types.utils import LiteLLMPydanticObjectBase


_MIN_PACE_RATIO = 1e-6
_MAX_PACE_FACTOR = 1e6
_MAX_PACE_EXPONENT = 8.0
_MIN_PACE_EXPONENT = -8.0


def _coerce_finite_float(value: Any, *, default: float) -> float:
    try:
        coerced_value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(coerced_value):
        return default
    return coerced_value


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def _stable_sigmoid(value: float) -> float:
    if not math.isfinite(value):
        return 0.0 if value < 0 else 1.0
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def compute_time_gate(
    *,
    hours_left: float,
    transition_center_hours: float,
    transition_width_hours: float,
) -> float:
    """Return a smooth 0..1 gate that increases as the deadline approaches."""
    safe_hours_left = max(_coerce_finite_float(hours_left, default=0.0), 0.0)
    safe_center = max(
        _coerce_finite_float(transition_center_hours, default=18.0),
        0.0,
    )
    safe_width = max(
        _coerce_finite_float(transition_width_hours, default=3.0),
        1e-9,
    )
    return _stable_sigmoid((safe_center - safe_hours_left) / safe_width)


def compute_pacing_factor(
    *,
    pace_ratio: float,
    gate: float,
    early_pace_exponent: float,
    late_pace_shift: float,
    late_time_bonus: float,
    min_pace_ratio: float,
    max_pacing_factor: float,
) -> float:
    """Blend early balance pressure and late burn-down pressure safely."""
    safe_gate = _clamp_float(_coerce_finite_float(gate, default=0.0), 0.0, 1.0)

    safe_pace_ratio = _coerce_finite_float(pace_ratio, default=1.0)
    if safe_pace_ratio < 0.0:
        safe_pace_ratio = 1.0
    safe_pace_ratio = max(
        safe_pace_ratio,
        max(_coerce_finite_float(min_pace_ratio, default=_MIN_PACE_RATIO), _MIN_PACE_RATIO),
    )

    safe_early_exponent = max(
        _coerce_finite_float(early_pace_exponent, default=1.4),
        0.0,
    )
    safe_late_shift = max(_coerce_finite_float(late_pace_shift, default=2.8), 0.0)
    safe_late_bonus = max(_coerce_finite_float(late_time_bonus, default=3.5), 0.0)
    safe_max_factor = max(
        _coerce_finite_float(max_pacing_factor, default=_MAX_PACE_FACTOR),
        1.0,
    )

    pace_exponent = safe_early_exponent - (safe_late_shift * safe_gate)
    pace_exponent = _clamp_float(pace_exponent, _MIN_PACE_EXPONENT, _MAX_PACE_EXPONENT)

    pace_factor = (safe_pace_ratio ** pace_exponent) * math.exp(safe_late_bonus * safe_gate)
    if not math.isfinite(pace_factor):
        return safe_max_factor
    return _clamp_float(pace_factor, 0.0, safe_max_factor)


def compute_base_weight_score(base_weight: float, base_weight_exponent: float) -> float:
    safe_base_weight = max(_coerce_finite_float(base_weight, default=1.0), 0.0)
    safe_base_exponent = max(_coerce_finite_float(base_weight_exponent, default=1.0), 0.0)
    if safe_base_weight <= 0.0:
        return 0.0
    return safe_base_weight ** safe_base_exponent


class RoutingArgs(LiteLLMPydanticObjectBase):
    # Smooth transition from balance-first behavior to burn-down behavior.
    base_weight_exponent: float = 1.6
    transition_center_hours: float = 18.0
    transition_width_hours: float = 3.0
    early_pace_exponent: float = 1.4
    late_pace_shift: float = 2.8
    late_time_bonus: float = 3.5
    min_time_ratio: float = 0.02
    min_pace_ratio: float = _MIN_PACE_RATIO
    max_pacing_factor: float = _MAX_PACE_FACTOR


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
        fast_mode_metrics: Optional[ChatGPTFastModeMetrics] = None,
    ):
        self.usage_service = usage_service
        self.routing_args = RoutingArgs(**(routing_args or {}))
        self.metrics = metrics or ChatGPTRoutingWeightMetrics()
        self.fast_mode_metrics = fast_mode_metrics or ChatGPTFastModeMetrics()

    @staticmethod
    def _get_request_metadata(
        request_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if request_kwargs is None:
            return None
        metadata_variable_name = get_metadata_variable_name_from_kwargs(request_kwargs)
        metadata = request_kwargs.get(metadata_variable_name)
        return metadata if isinstance(metadata, dict) else None

    @classmethod
    def _allows_fast_mode(
        cls,
        deployment: Dict[str, Any],
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return chatgpt_fast_mode_allowed(
            deployment.get("litellm_params"),
            metadata=cls._get_request_metadata(request_kwargs),
        )

    @staticmethod
    def _get_requested_service_tier(
        request_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        if request_kwargs is None:
            return None
        normalized = normalize_chatgpt_service_tier(request_kwargs.get("service_tier"))
        return normalized if isinstance(normalized, str) else None

    def _set_request_fast_mode_metadata(
        self,
        *,
        request_kwargs: Optional[Dict[str, Any]],
        deployment: Optional[Dict[str, Any]],
        requested_service_tier: Optional[str],
        effective_service_tier: Optional[str],
    ) -> None:
        if request_kwargs is None or deployment is None:
            return

        metadata_variable_name = get_metadata_variable_name_from_kwargs(request_kwargs)
        metadata = request_kwargs.get(metadata_variable_name)
        if not isinstance(metadata, dict):
            metadata = {}
            request_kwargs[metadata_variable_name] = metadata
        profile = self.usage_service.get_profile_name_from_deployment(deployment) or ""
        metadata.update(
            {
                "chatgpt_auth_profile": profile,
                "chatgpt_requested_service_tier": requested_service_tier or "default",
                "chatgpt_effective_service_tier": effective_service_tier or "default",
                "chatgpt_fast_mode_requested": requested_service_tier
                == CHATGPT_FAST_SERVICE_TIER,
                "chatgpt_fast_mode_effective": effective_service_tier
                == CHATGPT_FAST_SERVICE_TIER,
            }
        )
        allowance = chatgpt_fast_mode_allowance(
            deployment.get("litellm_params"), metadata=metadata
        )
        metadata.update(
            {
                "chatgpt_profile_allow_fast_mode": allowance["profile_allowed"],
                "chatgpt_virtual_key_allow_fast_mode": allowance[
                    "virtual_key_allowed"
                ],
                "chatgpt_fast_mode_allowed": allowance["allowed"],
            }
        )

    def _observe_fast_mode_request_metrics(
        self,
        *,
        deployment: Optional[Dict[str, Any]],
        request_kwargs: Optional[Dict[str, Any]],
        requested_service_tier: Optional[str],
        effective_service_tier: Optional[str],
    ) -> None:
        if deployment is None:
            return
        profile = self.usage_service.get_profile_name_from_deployment(deployment)
        if not profile:
            return
        self.fast_mode_metrics.observe_request(
            model_name=str(deployment.get("model_name", "") or ""),
            model_id=str(deployment.get("model_info", {}).get("id", "") or ""),
            profile=profile,
            virtual_key=self._get_virtual_key_label(request_kwargs),
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )

    @classmethod
    def _get_virtual_key_label(
        cls, request_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        metadata = cls._get_request_metadata(request_kwargs)
        if not isinstance(metadata, dict):
            return None
        return metadata.get("user_api_key_alias")

    @staticmethod
    def _should_raise_codex_usage_limit_error(blocked_reason: Optional[str]) -> bool:
        return blocked_reason in {
            "subscription_expired",
            "usage_profile_unavailable",
            "usage_window_exhausted",
        }

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
                coerced_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(coerced_value):
                continue
            return max(coerced_value, 0.0)
        return 1.0

    def _get_effective_weight_for_snapshot(
        self,
        *,
        base_weight: float,
        snapshot: Any,
        reference_time: float,
    ) -> Tuple[float, Optional[float], Optional[ChatGPTPacingInfo], Optional[str]]:
        effective_weight = compute_base_weight_score(
            base_weight,
            self.routing_args.base_weight_exponent,
        )
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
        if (
            pacing_info is None
            or pacing_info.time_ratio is None
            or pacing_info.pace_ratio is None
            or pacing_info.effective_deadline_at is None
        ):
            return effective_weight, None, pacing_info, None

        effective_deadline_at = _coerce_finite_float(
            pacing_info.effective_deadline_at,
            default=float("nan"),
        )
        if not math.isfinite(effective_deadline_at):
            return effective_weight, None, pacing_info, None

        safe_reference_time = _coerce_finite_float(reference_time, default=float("nan"))
        if not math.isfinite(safe_reference_time):
            return effective_weight, None, pacing_info, None

        hours_left = max((effective_deadline_at - safe_reference_time) / 3600.0, 0.0)
        transition_gate = compute_time_gate(
            hours_left=hours_left,
            transition_center_hours=self.routing_args.transition_center_hours,
            transition_width_hours=self.routing_args.transition_width_hours,
        )
        pacing_factor = compute_pacing_factor(
            pace_ratio=pacing_info.pace_ratio,
            gate=transition_gate,
            early_pace_exponent=self.routing_args.early_pace_exponent,
            late_pace_shift=self.routing_args.late_pace_shift,
            late_time_bonus=self.routing_args.late_time_bonus,
            min_pace_ratio=self.routing_args.min_pace_ratio,
            max_pacing_factor=self.routing_args.max_pacing_factor,
        )
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
        del messages, input
        self.usage_service.register_deployments(healthy_deployments)
        self.usage_service.ensure_background_refresh_task()
        reference_time = time.time()
        requested_service_tier = self._get_requested_service_tier(request_kwargs)
        requested_fast_mode = requested_service_tier == CHATGPT_FAST_SERVICE_TIER

        candidates_with_weights: List[Tuple[Dict[str, Any], float]] = []
        blocked_usage_results: List[Tuple[str, Any]] = []
        for deployment in healthy_deployments:
            profile = self.usage_service.get_profile_name_from_deployment(deployment)
            base_weight = self._get_base_weight(deployment)
            if not profile:
                candidates_with_weights.append(
                    (
                        deployment,
                        compute_base_weight_score(
                            base_weight,
                            self.routing_args.base_weight_exponent,
                        ),
                    )
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
                if self._should_raise_codex_usage_limit_error(blocked_reason):
                    blocked_usage_results.append((profile, snapshot.result))
            candidates_with_weights.append((deployment, effective_weight))

        selected_deployment: Optional[Dict[str, Any]] = None
        effective_service_tier = requested_service_tier or "default"

        if requested_fast_mode:
            fast_candidates = [
                (deployment, weight)
                for deployment, weight in candidates_with_weights
                if self._allows_fast_mode(deployment, request_kwargs)
            ]
            selected_deployment = self._choose_weighted_deployment(
                model_group=model_group,
                candidates_with_weights=fast_candidates,
            )
            if selected_deployment is None:
                selected_deployment = self._choose_weighted_deployment(
                    model_group=model_group,
                    candidates_with_weights=candidates_with_weights,
                )
                if (
                    selected_deployment is not None
                    and not self._allows_fast_mode(
                        selected_deployment, request_kwargs
                    )
                ):
                    effective_service_tier = "default"
        else:
            selected_deployment = self._choose_weighted_deployment(
                model_group=model_group,
                candidates_with_weights=candidates_with_weights,
            )

        if selected_deployment is None and blocked_usage_results:
            profile, result = blocked_usage_results[0]
            raise build_codex_rate_limit_error(profile=profile, result=result)

        self._set_request_fast_mode_metadata(
            request_kwargs=request_kwargs,
            deployment=selected_deployment,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        self._observe_fast_mode_request_metrics(
            deployment=selected_deployment,
            request_kwargs=request_kwargs,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        return selected_deployment

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Any] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        del messages, input
        self.usage_service.register_deployments(healthy_deployments)
        reference_time = time.time()
        requested_service_tier = self._get_requested_service_tier(request_kwargs)
        requested_fast_mode = requested_service_tier == CHATGPT_FAST_SERVICE_TIER

        candidates_with_weights: List[Tuple[Dict[str, Any], float]] = []
        blocked_usage_results: List[Tuple[str, Any]] = []
        for deployment in healthy_deployments:
            profile = self.usage_service.get_profile_name_from_deployment(deployment)
            base_weight = self._get_base_weight(deployment)
            if not profile:
                candidates_with_weights.append(
                    (
                        deployment,
                        compute_base_weight_score(
                            base_weight,
                            self.routing_args.base_weight_exponent,
                        ),
                    )
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
                if self._should_raise_codex_usage_limit_error(blocked_reason):
                    blocked_usage_results.append((profile, snapshot.result))
            candidates_with_weights.append((deployment, effective_weight))

        selected_deployment: Optional[Dict[str, Any]] = None
        effective_service_tier = requested_service_tier or "default"

        if requested_fast_mode:
            fast_candidates = [
                (deployment, weight)
                for deployment, weight in candidates_with_weights
                if self._allows_fast_mode(deployment, request_kwargs)
            ]
            selected_deployment = self._choose_weighted_deployment(
                model_group=model_group,
                candidates_with_weights=fast_candidates,
            )
            if selected_deployment is None:
                selected_deployment = self._choose_weighted_deployment(
                    model_group=model_group,
                    candidates_with_weights=candidates_with_weights,
                )
                if (
                    selected_deployment is not None
                    and not self._allows_fast_mode(
                        selected_deployment, request_kwargs
                    )
                ):
                    effective_service_tier = "default"
        else:
            selected_deployment = self._choose_weighted_deployment(
                model_group=model_group,
                candidates_with_weights=candidates_with_weights,
            )

        if selected_deployment is None and blocked_usage_results:
            profile, result = blocked_usage_results[0]
            raise build_codex_rate_limit_error(profile=profile, result=result)

        self._set_request_fast_mode_metadata(
            request_kwargs=request_kwargs,
            deployment=selected_deployment,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        self._observe_fast_mode_request_metrics(
            deployment=selected_deployment,
            request_kwargs=request_kwargs,
            requested_service_tier=requested_service_tier,
            effective_service_tier=effective_service_tier,
        )
        return selected_deployment
