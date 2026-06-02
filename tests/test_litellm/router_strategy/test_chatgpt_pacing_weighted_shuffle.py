from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from litellm.llms.chatgpt.usage_service import ChatGPTUsageSnapshot, UsageResult, UsageWindow
from litellm.router_strategy.chatgpt_pacing_weighted_shuffle import (
    ChatGPTRoutingWeightMetrics,
    ChatGPTPacingWeightedShuffle,
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
