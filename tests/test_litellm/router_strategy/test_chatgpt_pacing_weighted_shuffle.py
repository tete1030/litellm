import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from litellm.llms.chatgpt.usage_service import ChatGPTUsageSnapshot, UsageResult, UsageWindow
from litellm.router_strategy.chatgpt_pacing_weighted_shuffle import (
    ChatGPTRoutingWeightMetrics,
    ChatGPTPacingWeightedShuffle,
    compute_base_weight_score,
    compute_pacing_factor,
    compute_time_gate,
)


def _make_snapshot(
    *,
    profile: str,
    used_percent: float,
    reset_at: int,
    subscription_expires_at: int | None = None,
    effective_available: bool = True,
) -> ChatGPTUsageSnapshot:
    return ChatGPTUsageSnapshot(
        profile=profile,
        refreshed_at=1700000000.0,
        result=UsageResult(
            profile=profile,
            account_id=f"acct-{profile}",
            plan="plus",
            account_type="plus",
            credits_balance=12.0,
            status="ok",
            windows=[
                UsageWindow(
                    label="1w",
                    used_percent=used_percent,
                    reset_at=reset_at,
                    limit_seconds=604800,
                )
            ],
            has_active_subscription=True,
            subscription_expires_at=subscription_expires_at,
            effective_available=effective_available,
        ),
    )


def _make_chatgpt_deployment(profile: str, *, weight: float = 1.0) -> dict:
    return {
        "model_name": "gpt-5.4",
        "litellm_params": {
            "model": "chatgpt/gpt-5.4",
            "chatgpt_auth_profile": profile,
            "weight": weight,
        },
        "model_info": {"id": f"chatgpt-{profile}"},
    }


def test_time_gate_is_smooth_and_bounded() -> None:
    early_gate = compute_time_gate(
        hours_left=90.0,
        transition_center_hours=18.0,
        transition_width_hours=3.0,
    )
    midpoint_gate = compute_time_gate(
        hours_left=18.0,
        transition_center_hours=18.0,
        transition_width_hours=3.0,
    )
    late_gate = compute_time_gate(
        hours_left=12.0,
        transition_center_hours=18.0,
        transition_width_hours=3.0,
    )

    assert 0.0 <= early_gate < 0.01
    assert midpoint_gate == pytest.approx(0.5)
    assert late_gate > 0.85
    assert late_gate < 1.0


def test_pacing_factor_flips_direction_late_and_handles_invalid_inputs() -> None:
    early_gate = compute_time_gate(
        hours_left=90.0,
        transition_center_hours=18.0,
        transition_width_hours=3.0,
    )
    late_gate = compute_time_gate(
        hours_left=12.0,
        transition_center_hours=18.0,
        transition_width_hours=3.0,
    )

    early_underuse = compute_pacing_factor(
        pace_ratio=1.5,
        gate=early_gate,
        early_pace_exponent=1.4,
        late_pace_shift=2.8,
        late_time_bonus=3.5,
        min_pace_ratio=1e-6,
        max_pacing_factor=1e6,
    )
    early_overuse = compute_pacing_factor(
        pace_ratio=0.6,
        gate=early_gate,
        early_pace_exponent=1.4,
        late_pace_shift=2.8,
        late_time_bonus=3.5,
        min_pace_ratio=1e-6,
        max_pacing_factor=1e6,
    )
    late_underuse = compute_pacing_factor(
        pace_ratio=1.5,
        gate=late_gate,
        early_pace_exponent=1.4,
        late_pace_shift=2.8,
        late_time_bonus=3.5,
        min_pace_ratio=1e-6,
        max_pacing_factor=1e6,
    )
    late_overuse = compute_pacing_factor(
        pace_ratio=0.6,
        gate=late_gate,
        early_pace_exponent=1.4,
        late_pace_shift=2.8,
        late_time_bonus=3.5,
        min_pace_ratio=1e-6,
        max_pacing_factor=1e6,
    )

    assert early_underuse > early_overuse
    assert late_overuse > late_underuse
    assert math.isfinite(
        compute_pacing_factor(
            pace_ratio=0.0,
            gate=late_gate,
            early_pace_exponent=1.4,
            late_pace_shift=2.8,
            late_time_bonus=3.5,
            min_pace_ratio=1e-6,
            max_pacing_factor=1e6,
        )
    )
    assert compute_pacing_factor(
        pace_ratio=-2.0,
        gate=late_gate,
        early_pace_exponent=1.4,
        late_pace_shift=2.8,
        late_time_bonus=3.5,
        min_pace_ratio=1e-6,
        max_pacing_factor=1e6,
    ) == pytest.approx(
        compute_pacing_factor(
            pace_ratio=1.0,
            gate=late_gate,
            early_pace_exponent=1.4,
            late_pace_shift=2.8,
            late_time_bonus=3.5,
            min_pace_ratio=1e-6,
            max_pacing_factor=1e6,
        )
    )


def test_base_weight_score_rejects_invalid_inputs() -> None:
    assert compute_base_weight_score(0.0, 1.6) == 0.0
    assert compute_base_weight_score(-5.0, 1.6) == 0.0
    assert compute_base_weight_score(float("nan"), 1.6) == 1.0
    assert compute_base_weight_score(20.0, float("nan")) == 20.0


def test_sync_strategy_prefers_near_deadline_account_and_records_metrics() -> None:
    registry = CollectorRegistry()
    metrics = ChatGPTRoutingWeightMetrics(registry=registry)
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot_sync.side_effect = lambda profile: {
        "urgent": _make_snapshot(profile="urgent", used_percent=20.0, reset_at=1700100000),
        "steady": _make_snapshot(profile="steady", used_percent=20.0, reset_at=1700500000),
    }[profile]

    strategy = ChatGPTPacingWeightedShuffle(
        usage_service=usage_service,
        metrics=metrics,
    )
    healthy_deployments = [
        _make_chatgpt_deployment("urgent"),
        _make_chatgpt_deployment("steady"),
    ]

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ) as mock_choices:
        selected = strategy.get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=healthy_deployments,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "chatgpt-urgent"
    weights = mock_choices.call_args.kwargs["weights"]
    assert weights[0] > weights[1]
    assert registry.get_sample_value(
        "litellm_chatgpt_routing_base_weight",
        {"model_name": "gpt-5.4", "model_id": "chatgpt-urgent", "profile": "urgent"},
    ) == 1.0
    assert registry.get_sample_value(
        "litellm_chatgpt_routing_weekly_remaining_ratio",
        {"model_name": "gpt-5.4", "model_id": "chatgpt-urgent", "profile": "urgent"},
    ) == 0.8
    assert registry.get_sample_value(
        "litellm_chatgpt_routing_effective_deadline_timestamp_seconds",
        {"model_name": "gpt-5.4", "model_id": "chatgpt-urgent", "profile": "urgent"},
    ) == 1700100000.0
    assert registry.get_sample_value(
        "litellm_chatgpt_routing_effective_weight",
        {"model_name": "gpt-5.4", "model_id": "chatgpt-urgent", "profile": "urgent"},
    ) == weights[0]


def test_sync_strategy_sets_effective_weight_zero_for_blocked_profile() -> None:
    registry = CollectorRegistry()
    metrics = ChatGPTRoutingWeightMetrics(registry=registry)
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot_sync.side_effect = lambda profile: {
        "blocked": _make_snapshot(profile="blocked", used_percent=100.0, reset_at=1700100000),
        "healthy": _make_snapshot(profile="healthy", used_percent=30.0, reset_at=1700300000),
    }[profile]

    strategy = ChatGPTPacingWeightedShuffle(
        usage_service=usage_service,
        metrics=metrics,
    )

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ):
        selected = strategy.get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=[
                _make_chatgpt_deployment("blocked"),
                _make_chatgpt_deployment("healthy"),
            ],
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "chatgpt-healthy"
    assert registry.get_sample_value(
        "litellm_chatgpt_routing_effective_weight",
        {"model_name": "gpt-5.4", "model_id": "chatgpt-blocked", "profile": "blocked"},
    ) == 0.0


@pytest.mark.asyncio
async def test_async_strategy_uses_async_snapshots() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot = AsyncMock(
        side_effect=lambda profile: {
            "buy1": _make_snapshot(profile="buy1", used_percent=25.0, reset_at=1700150000),
            "buy2": _make_snapshot(profile="buy2", used_percent=60.0, reset_at=1700400000),
        }[profile]
    )
    usage_service.register_deployments.return_value = None
    usage_service.ensure_background_refresh_task.return_value = None

    strategy = ChatGPTPacingWeightedShuffle(usage_service=usage_service)
    healthy_deployments = [
        _make_chatgpt_deployment("buy1"),
        _make_chatgpt_deployment("buy2"),
    ]

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ) as mock_choices:
        selected = await strategy.async_get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=healthy_deployments,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "chatgpt-buy1"
    assert usage_service.get_snapshot.await_count == 2
    weights = mock_choices.call_args.kwargs["weights"]
    assert weights[0] > weights[1]


def test_sync_strategy_prefers_fast_enabled_profiles_for_fast_requests() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot_sync.side_effect = lambda profile: {
        "fast-ok": _make_snapshot(profile="fast-ok", used_percent=20.0, reset_at=1700100000),
        "slow-only": _make_snapshot(profile="slow-only", used_percent=20.0, reset_at=1700100000),
    }[profile]

    strategy = ChatGPTPacingWeightedShuffle(usage_service=usage_service)
    slow_only = _make_chatgpt_deployment("slow-only")
    healthy_deployments = [
        _make_chatgpt_deployment("fast-ok"),
        {
            **slow_only,
            "litellm_params": {
                **slow_only["litellm_params"],
                "chatgpt_allow_fast_mode": False,
            },
        },
    ]
    request_kwargs = {"service_tier": "fast", "metadata": {}}

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ) as mock_choices:
        selected = strategy.get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "chatgpt-fast-ok"
    assert len(mock_choices.call_args.kwargs["weights"]) == 1
    assert request_kwargs["metadata"]["chatgpt_auth_profile"] == "fast-ok"
    assert request_kwargs["metadata"]["chatgpt_requested_service_tier"] == "priority"
    assert request_kwargs["metadata"]["chatgpt_effective_service_tier"] == "priority"
    assert request_kwargs["metadata"]["chatgpt_profile_allow_fast_mode"] is True
    assert request_kwargs["metadata"]["chatgpt_virtual_key_allow_fast_mode"] is True
    assert request_kwargs["metadata"]["chatgpt_fast_mode_allowed"] is True


def test_sync_strategy_falls_back_to_restricted_profiles_with_fast_mode_disabled() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot_sync.return_value = _make_snapshot(
        profile="slow-only", used_percent=20.0, reset_at=1700100000
    )

    strategy = ChatGPTPacingWeightedShuffle(usage_service=usage_service)
    healthy_deployments = [
        {
            **_make_chatgpt_deployment("slow-only"),
            "litellm_params": {
                **_make_chatgpt_deployment("slow-only")["litellm_params"],
                "chatgpt_allow_fast_mode": False,
            },
        }
    ]
    request_kwargs = {"service_tier": "priority", "metadata": {}}

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ):
        selected = strategy.get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
        )

    assert selected is not None
    assert selected["model_info"]["id"] == "chatgpt-slow-only"
    assert request_kwargs["metadata"]["chatgpt_auth_profile"] == "slow-only"
    assert request_kwargs["metadata"]["chatgpt_requested_service_tier"] == "priority"
    assert request_kwargs["metadata"]["chatgpt_effective_service_tier"] == "default"
    assert request_kwargs["metadata"]["chatgpt_profile_allow_fast_mode"] is False
    assert request_kwargs["metadata"]["chatgpt_virtual_key_allow_fast_mode"] is True
    assert request_kwargs["metadata"]["chatgpt_fast_mode_allowed"] is False


@pytest.mark.parametrize(
    (
        "metadata",
        "default_allow",
        "expected_effective_tier",
        "expected_virtual_key_allow",
        "expected_allowed",
    ),
    [
        (
            {
                "user_api_key_alias": "vk-denied",
                "user_api_key_auth_metadata": {
                    "chatgpt_virtual_key_allow_fast_mode": False
                },
            },
            True,
            "default",
            False,
            False,
        ),
        (
            {"user_api_key_alias": "vk-default-denied"},
            False,
            "default",
            False,
            False,
        ),
        (
            {
                "user_api_key_alias": "vk-allowed",
                "user_api_key_auth_metadata": {
                    "chatgpt_virtual_key_allow_fast_mode": True
                },
            },
            False,
            "priority",
            True,
            True,
        ),
    ],
)
def test_sync_strategy_applies_virtual_key_fast_mode_allowance(
    metadata: dict,
    default_allow: bool,
    expected_effective_tier: str,
    expected_virtual_key_allow: bool,
    expected_allowed: bool,
) -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.get_snapshot_sync.return_value = _make_snapshot(
        profile="fast-ok", used_percent=20.0, reset_at=1700100000
    )

    fast_mode_metrics = MagicMock()
    strategy = ChatGPTPacingWeightedShuffle(
        usage_service=usage_service, fast_mode_metrics=fast_mode_metrics
    )
    request_kwargs = {"service_tier": "priority", "metadata": metadata}

    with patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.time.time",
        return_value=1700000000,
    ), patch(
        "litellm.router_strategy.chatgpt_pacing_weighted_shuffle.random.choices",
        return_value=[0],
    ), patch(
        "litellm.chatgpt_virtual_key_allow_fast_mode_default",
        default_allow,
        create=True,
    ):
        selected = strategy.get_available_deployments(
            model_group="gpt-5.4",
            healthy_deployments=[_make_chatgpt_deployment("fast-ok")],
            request_kwargs=request_kwargs,
        )

    assert selected is not None
    assert request_kwargs["metadata"]["chatgpt_effective_service_tier"] == (
        expected_effective_tier
    )
    assert request_kwargs["metadata"]["chatgpt_profile_allow_fast_mode"] is True
    assert request_kwargs["metadata"]["chatgpt_virtual_key_allow_fast_mode"] is (
        expected_virtual_key_allow
    )
    assert request_kwargs["metadata"]["chatgpt_fast_mode_allowed"] is expected_allowed
    fast_mode_metrics.observe_request.assert_called_once_with(
        model_name="gpt-5.4",
        model_id="chatgpt-fast-ok",
        profile="fast-ok",
        virtual_key=metadata.get("user_api_key_alias"),
        requested_service_tier="priority",
        effective_service_tier=expected_effective_tier,
    )
