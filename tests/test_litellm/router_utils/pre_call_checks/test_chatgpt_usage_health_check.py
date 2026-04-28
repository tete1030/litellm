from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.llms.chatgpt.usage_service import ChatGPTUsageSnapshot, UsageResult, UsageWindow
from litellm.router_utils.pre_call_checks.chatgpt_usage_health_check import (
    ChatGPTUsageHealthCheck,
)
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)


def test_evaluate_usage_result_blocks_when_any_window_reaches_100_percent() -> None:
    usage_result = UsageResult(
        profile="buy4-bus",
        account_id="acct_123",
        plan="team",
        credits_balance=10.0,
        status="ok",
        windows=[
            UsageWindow(label="5h", used_percent=100.0, reset_at=1700000000),
            UsageWindow(label="1w", used_percent=62.0, reset_at=1700600000),
        ],
    )

    decision = ChatGPTUsageHealthCheck.evaluate_usage_result(usage_result)

    assert decision.action == "block"
    assert decision.reason == "usage_window_exhausted"
    assert decision.blocked_window_labels == ("5h",)


def test_evaluate_usage_result_allows_below_100_percent_windows() -> None:
    usage_result = UsageResult(
        profile="buy4-bus",
        account_id="acct_123",
        plan="team",
        credits_balance=10.0,
        status="ok",
        windows=[
            UsageWindow(label="5h", used_percent=99.0, reset_at=1700000000),
            UsageWindow(label="1w", used_percent=100.0 - 1e-6, reset_at=1700600000),
        ],
    )

    decision = ChatGPTUsageHealthCheck.evaluate_usage_result(usage_result)

    assert decision.action == "allow"


def test_evaluate_usage_result_blocks_deactivated_workspace_error() -> None:
    usage_result = UsageResult(
        profile="buy3-bus",
        account_id="acct_123",
        plan="team",
        credits_balance=None,
        status="error",
        windows=[],
        error='usage request failed (402): {"detail":{"code":"deactivated_workspace"}}',
    )

    decision = ChatGPTUsageHealthCheck.evaluate_usage_result(usage_result)

    assert decision.action == "block"
    assert decision.reason == "usage_endpoint_unavailable"


def test_evaluate_usage_result_blocks_auth_errors() -> None:
    usage_result = UsageResult(
        profile="buy4-bus",
        account_id="acct_123",
        plan="team",
        credits_balance=None,
        status="error",
        windows=[],
        error=(
            "usage auth failed: Refresh token failed for profile 'buy4-bus': "
            "Client error '401 Unauthorized' for url 'https://auth.openai.com/oauth/token'"
        ),
    )

    decision = ChatGPTUsageHealthCheck.evaluate_usage_result(usage_result)

    assert decision.action == "block"
    assert decision.reason == "usage_auth_unavailable"


def test_evaluate_usage_result_allows_non_blocking_usage_fetch_errors() -> None:
    usage_result = UsageResult(
        profile="buy3-bus",
        account_id="acct_123",
        plan="team",
        credits_balance=None,
        status="error",
        windows=[],
        error="usage request failed (500): upstream timeout",
    )

    decision = ChatGPTUsageHealthCheck.evaluate_usage_result(usage_result)

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_async_filter_deployments_blocks_only_exhausted_chatgpt_profiles() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.register_deployments.return_value = None
    usage_service.ensure_background_refresh_task.return_value = None
    usage_service.get_snapshot = AsyncMock()
    usage_service.get_snapshot.side_effect = [
        ChatGPTUsageSnapshot(
            profile="buy3-bus",
            refreshed_at=1700000000.0,
            result=UsageResult(
                profile="buy3-bus",
                account_id="acct_blocked",
                plan="team",
                credits_balance=0.0,
                status="ok",
                windows=[UsageWindow(label="1w", used_percent=100.0, reset_at=1700600000)],
            ),
        ),
        ChatGPTUsageSnapshot(
            profile="my",
            refreshed_at=1700000000.0,
            result=UsageResult(
                profile="my",
                account_id="acct_ok",
                plan="team",
                credits_balance=12.0,
                status="ok",
                windows=[UsageWindow(label="5h", used_percent=20.0, reset_at=1700003600)],
            ),
        ),
    ]
    callback = ChatGPTUsageHealthCheck(usage_service=usage_service)

    healthy_deployments = [
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "buy3-bus"},
            "model_info": {"id": "chatgpt-buy3-bus-codex"},
        },
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "my"},
            "model_info": {"id": "chatgpt-my-codex"},
        },
    ]

    filtered = await callback.async_filter_deployments(
        model="gpt-5.3-codex",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=None,
        parent_otel_span=None,
    )

    assert [item["model_info"]["id"] for item in filtered] == ["chatgpt-my-codex"]


@pytest.mark.asyncio
async def test_async_filter_deployments_blocks_usage_endpoint_deactivated_workspace() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.register_deployments.return_value = None
    usage_service.ensure_background_refresh_task.return_value = None
    usage_service.get_snapshot = AsyncMock(
        side_effect=[
            ChatGPTUsageSnapshot(
                profile="buy3-bus",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="buy3-bus",
                    account_id="acct_blocked",
                    plan="team",
                    credits_balance=None,
                    status="error",
                    windows=[],
                    error='usage request failed (402): {"detail":{"code":"deactivated_workspace"}}',
                ),
            ),
            ChatGPTUsageSnapshot(
                profile="my",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="my",
                    account_id="acct_ok",
                    plan="team",
                    credits_balance=12.0,
                    status="ok",
                    windows=[UsageWindow(label="5h", used_percent=20.0, reset_at=1700003600)],
                ),
            ),
        ]
    )
    callback = ChatGPTUsageHealthCheck(usage_service=usage_service)

    healthy_deployments = [
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "buy3-bus"},
            "model_info": {"id": "chatgpt-buy3-bus-codex"},
        },
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "my"},
            "model_info": {"id": "chatgpt-my-codex"},
        },
    ]

    filtered = await callback.async_filter_deployments(
        model="gpt-5.3-codex",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=None,
        parent_otel_span=None,
    )

    assert [item["model_info"]["id"] for item in filtered] == ["chatgpt-my-codex"]


@pytest.mark.asyncio
async def test_async_filter_deployments_returns_empty_when_all_chatgpt_profiles_are_exhausted() -> None:
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.register_deployments.return_value = None
    usage_service.ensure_background_refresh_task.return_value = None
    usage_service.get_snapshot = AsyncMock(
        side_effect=[
            ChatGPTUsageSnapshot(
                profile="buy3-bus",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="buy3-bus",
                    account_id="acct_1",
                    plan="team",
                    credits_balance=0.0,
                    status="ok",
                    windows=[UsageWindow(label="5h", used_percent=100.0, reset_at=1700003600)],
                ),
            ),
            ChatGPTUsageSnapshot(
                profile="buy4-bus",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="buy4-bus",
                    account_id="acct_2",
                    plan="team",
                    credits_balance=0.0,
                    status="ok",
                    windows=[UsageWindow(label="1w", used_percent=100.0, reset_at=1700600000)],
                ),
            ),
        ]
    )
    callback = ChatGPTUsageHealthCheck(usage_service=usage_service)

    healthy_deployments = [
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "buy3-bus"},
            "model_info": {"id": "chatgpt-buy3-bus-codex"},
        },
        {
            "model_name": "gpt-5.3-codex",
            "litellm_params": {"model": "chatgpt/gpt-5.3-codex", "chatgpt_auth_profile": "buy4-bus"},
            "model_info": {"id": "chatgpt-buy4-bus-codex"},
        },
    ]

    filtered = await callback.async_filter_deployments(
        model="gpt-5.3-codex",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=None,
        parent_otel_span=None,
    )

    assert filtered == []


def test_router_registers_chatgpt_usage_health_check() -> None:
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.3-codex",
                "litellm_params": {
                    "model": "chatgpt/gpt-5.3-codex",
                    "chatgpt_auth_profile": "buy3-bus",
                },
                "model_info": {"id": "chatgpt-buy3-bus-codex"},
            }
        ],
        optional_pre_call_checks=["chatgpt_usage_health_check"],
    )

    assert router.optional_callbacks is not None
    assert any(
        isinstance(callback, ChatGPTUsageHealthCheck)
        for callback in router.optional_callbacks
    )


@pytest.mark.asyncio
async def test_usage_health_check_runs_before_session_affinity_when_pinned_deployment_is_blocked() -> None:
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.4",
                "litellm_params": {
                    "model": "chatgpt/gpt-5.4",
                    "chatgpt_auth_profile": "my",
                },
                "model_info": {"id": "chatgpt-my"},
            },
            {
                "model_name": "gpt-5.4",
                "litellm_params": {
                    "model": "chatgpt/gpt-5.4",
                    "chatgpt_auth_profile": "buy7",
                },
                "model_info": {"id": "chatgpt-buy7"},
            },
        ],
        optional_pre_call_checks=["session_affinity", "chatgpt_usage_health_check"],
    )

    assert router.optional_callbacks is not None

    usage_callback = next(
        callback
        for callback in router.optional_callbacks
        if isinstance(callback, ChatGPTUsageHealthCheck)
    )
    usage_service = MagicMock()
    usage_service.get_profile_name_from_deployment.side_effect = lambda deployment: deployment[
        "litellm_params"
    ].get("chatgpt_auth_profile")
    usage_service.register_deployments.return_value = None
    usage_service.ensure_background_refresh_task.return_value = None
    usage_service.get_snapshot = AsyncMock(
        side_effect=[
            ChatGPTUsageSnapshot(
                profile="my",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="my",
                    account_id="acct_blocked",
                    plan="plus",
                    credits_balance=1.0,
                    status="ok",
                    windows=[
                        UsageWindow(
                            label="secondary_window",
                            used_percent=100.0,
                            reset_at=1700600000,
                        )
                    ],
                ),
            ),
            ChatGPTUsageSnapshot(
                profile="buy7",
                refreshed_at=1700000000.0,
                result=UsageResult(
                    profile="buy7",
                    account_id="acct_ok",
                    plan="plus",
                    credits_balance=12.0,
                    status="ok",
                    windows=[
                        UsageWindow(
                            label="secondary_window",
                            used_percent=39.0,
                            reset_at=1700600000,
                        )
                    ],
                ),
            ),
        ]
    )
    usage_callback.usage_service = usage_service

    session_id = "sticky-session"
    session_cache_key = DeploymentAffinityCheck.get_session_affinity_cache_key(
        model_group="gpt-5.4", session_id=session_id
    )
    await router.cache.async_set_cache(
        key=session_cache_key,
        value={"id": "chatgpt-my"},
        ttl=60,
    )

    filtered = await router.async_callback_filter_deployments(
        model="gpt-5.4",
        healthy_deployments=list(router.model_list),
        messages=[],
        request_kwargs={"metadata": {"session_id": session_id}},
        parent_otel_span=None,
    )

    assert [deployment["model_info"]["id"] for deployment in filtered] == [
        "chatgpt-buy7"
    ]
