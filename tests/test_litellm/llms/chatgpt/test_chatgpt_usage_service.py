from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.chatgpt.common_utils import RefreshAccessTokenError
from litellm.llms.chatgpt.usage_service import (
    build_codex_rate_limit_error,
    compute_chatgpt_pacing_info,
    consume_rate_limit_reset_credit_for_profile,
    _extract_account_metadata,
    fetch_usage_for_profile,
    fetch_rate_limit_reset_credits_for_profile,
    get_effective_deadline_at,
    get_weekly_pacing_window,
    is_usage_result_expired,
    normalize_usage_payload,
)


def test_fetch_usage_for_profile_returns_error_result_on_auth_failure() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy4"

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(
            authenticator,
            {"account_id": "acct_buy4", "refresh_token": "***"},
        ),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        side_effect=RefreshAccessTokenError(
            status_code=401,
            message=(
                "Refresh token failed for profile 'buy4-bus': "
                "Client error '401 Unauthorized'"
            ),
        ),
    ):
        result = fetch_usage_for_profile("buy4-bus")

    assert result.profile == "buy4-bus"
    assert result.account_id == "acct_buy4"
    assert result.status == "error"
    assert result.effective_available is False
    assert (
        result.error
        == "usage auth failed: Refresh token failed for profile 'buy4-bus': Client error '401 Unauthorized'"
    )


def test_normalize_usage_payload_free_account_is_unavailable() -> None:
    result = normalize_usage_payload(
        profile="buy1",
        account_id="acct-buy1",
        payload={
            "plan_type": "free",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
            },
        },
    )

    assert result.account_type == "free"
    assert result.effective_available is False


def test_extract_account_metadata_parses_accounts_check_v4_payload() -> None:
    payload = {
        "accounts": {
            "default": {
                "account": {"plan_type": "prolite"},
                "entitlement": {
                    "has_active_subscription": True,
                    "expires_at": "2026-05-20T08:50:24+00:00",
                    "renews_at": "2026-05-19T07:50:24+00:00",
                },
            }
        }
    }

    account_type, has_active_subscription, expires_at, renews_at = _extract_account_metadata(
        payload, account_id=""
    )

    assert account_type == "prolite"
    assert has_active_subscription is True
    assert expires_at == 1779267024
    assert renews_at == 1779177024


def test_fetch_usage_for_profile_applies_accounts_metadata_to_availability() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy10"

    usage_response = MagicMock()
    usage_response.raise_for_status.return_value = None
    usage_response.json.return_value = {
        "plan_type": "plus",
        "rate_limit": {
            "allowed": True,
            "limit_reached": False,
        },
    }

    client = MagicMock()
    client.get.return_value = usage_response

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(authenticator, {"account_id": "acct_buy10", "access_token": "token"}),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        return_value="token",
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_httpx_client",
        return_value=client,
    ), patch(
        "litellm.llms.chatgpt.usage_service._fetch_account_metadata",
        return_value=("free", True, 1779267024, 1779177024),
    ):
        result = fetch_usage_for_profile("buy10")

    assert result.status == "ok"
    assert result.account_type == "free"
    assert result.subscription_expires_at == 1779267024
    assert result.subscription_renews_at == 1779177024
    assert result.effective_available is False


def test_fetch_rate_limit_reset_credits_for_profile_returns_error_result_on_auth_failure() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy4"

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(
            authenticator,
            {"account_id": "acct_buy4", "refresh_token": "***"},
        ),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        side_effect=RefreshAccessTokenError(
            status_code=401,
            message=(
                "Refresh token failed for profile 'buy4-bus': "
                "Client error '401 Unauthorized'"
            ),
        ),
    ):
        result = fetch_rate_limit_reset_credits_for_profile("buy4-bus")

    assert result.profile == "buy4-bus"
    assert result.account_id == "acct_buy4"
    assert result.status == "error"
    assert result.available_count is None
    assert result.credits == []
    assert (
        result.error
        == "reset credits auth failed: Refresh token failed for profile 'buy4-bus': Client error '401 Unauthorized'"
    )


def test_fetch_rate_limit_reset_credits_for_profile_parses_available_count_and_credits() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy10"

    reset_response = MagicMock()
    reset_response.raise_for_status.return_value = None
    reset_response.json.return_value = {
        "available_count": 1,
        "credits": [
            {
                "credit_id": "credit-123",
                "title": "Weekly reset",
                "available": True,
            }
        ],
    }

    client = MagicMock()
    client.get.return_value = reset_response

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(authenticator, {"account_id": "acct_buy10", "access_token": "token"}),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        return_value="token",
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_httpx_client",
        return_value=client,
    ):
        result = fetch_rate_limit_reset_credits_for_profile("buy10")

    assert result.status == "ok"
    assert result.available_count == 1
    assert result.code is None
    assert result.credits[0]["credit_id"] == "credit-123"
    assert result.credits[0]["title"] == "Weekly reset"
    client.get.assert_called_once()
    request_headers = client.get.call_args.kwargs["headers"]
    assert request_headers["Authorization"] == "Bearer token"
    assert request_headers["ChatGPT-Account-Id"] == "acct_buy10"


def test_consume_rate_limit_reset_credit_for_profile_posts_expected_payload() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy10"

    consume_response = MagicMock()
    consume_response.raise_for_status.return_value = None
    consume_response.json.return_value = {
        "code": "reset",
        "available_count": 0,
        "credits": [],
    }

    client = MagicMock()
    client.post.return_value = consume_response

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(authenticator, {"account_id": "acct_buy10", "access_token": "token"}),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        return_value="token",
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_httpx_client",
        return_value=client,
    ):
        result = consume_rate_limit_reset_credit_for_profile(
            "buy10",
            "credit-123",
            "redeem-456",
        )

    assert result.status == "ok"
    assert result.code == "reset"
    assert result.available_count == 0
    client.post.assert_called_once()
    request_kwargs = client.post.call_args.kwargs
    assert request_kwargs["json"] == {
        "credit_id": "credit-123",
        "redeem_request_id": "redeem-456",
    }
    assert request_kwargs["headers"]["Authorization"] == "Bearer token"
    assert request_kwargs["headers"]["Content-Type"] == "application/json"


def test_consume_rate_limit_reset_credit_for_profile_marks_business_error_code_as_error() -> None:
    authenticator = MagicMock()
    authenticator.get_account_id.return_value = "acct_buy10"

    consume_response = MagicMock()
    consume_response.raise_for_status.return_value = None
    consume_response.json.return_value = {
        "code": "already_redeemed",
        "message": "This reset credit has already been redeemed.",
        "available_count": 0,
        "credits": [],
    }

    client = MagicMock()
    client.post.return_value = consume_response

    with patch(
        "litellm.llms.chatgpt.usage_service._load_auth_data",
        return_value=(authenticator, {"account_id": "acct_buy10", "access_token": "token"}),
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_usage_access_token",
        return_value="token",
    ), patch(
        "litellm.llms.chatgpt.usage_service._get_httpx_client",
        return_value=client,
    ):
        result = consume_rate_limit_reset_credit_for_profile(
            "buy10",
            "credit-123",
            "redeem-456",
        )

    assert result.status == "error"
    assert result.code == "already_redeemed"
    assert result.message == "This reset credit has already been redeemed."
    assert result.error == "This reset credit has already been redeemed."


def test_compute_chatgpt_pacing_info_prefers_weekly_window_and_earliest_deadline() -> None:
    result = normalize_usage_payload(
        profile="buy2",
        account_id="acct-buy2",
        payload={
            "plan_type": "plus",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
                "primary_window": {
                    "limit_window_seconds": 18000,
                    "used_percent": 25,
                    "reset_at": 1700003600,
                },
                "secondary_window": {
                    "limit_window_seconds": 604800,
                    "used_percent": 80,
                    "reset_at": 1700600000,
                },
            },
        },
    )
    result.subscription_expires_at = 1700300000

    weekly_window = get_weekly_pacing_window(result)
    assert weekly_window is not None
    assert weekly_window.label == "1w"
    assert get_effective_deadline_at(result, weekly_window) == 1700300000

    pacing_info = compute_chatgpt_pacing_info(
        result,
        now=1700000000,
        min_time_ratio=0.02,
    )

    assert pacing_info is not None
    assert pacing_info.remaining_ratio == pytest.approx(0.2)
    assert pacing_info.effective_deadline_at == 1700300000
    assert pacing_info.time_ratio == pytest.approx(300000 / 604800)
    assert pacing_info.pace_ratio == pytest.approx(0.2 / (300000 / 604800))


def test_compute_chatgpt_pacing_info_falls_back_to_longest_window() -> None:
    result = normalize_usage_payload(
        profile="buy3",
        account_id="acct-buy3",
        payload={
            "plan_type": "plus",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
                "primary_window": {
                    "limit_window_seconds": 43200,
                    "used_percent": 10,
                    "reset_at": 1700043200,
                },
                "secondary_window": {
                    "limit_window_seconds": 259200,
                    "used_percent": 40,
                    "reset_at": 1700259200,
                },
            },
        },
    )

    pacing_info = compute_chatgpt_pacing_info(result, now=1700000000)

    assert pacing_info is not None
    assert pacing_info.window_label == "3d"
    assert pacing_info.remaining_ratio == pytest.approx(0.6)
    assert pacing_info.effective_deadline_at == 1700259200


def test_is_usage_result_expired_detects_past_subscription_deadline() -> None:
    result = normalize_usage_payload(
        profile="buy4",
        account_id="acct-buy4",
        payload={
            "plan_type": "plus",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
            },
        },
    )
    result.subscription_expires_at = 1699999999

    assert is_usage_result_expired(result, now=1700000000) is True


def test_build_codex_rate_limit_error_for_exhausted_weekly_window() -> None:
    result = normalize_usage_payload(
        profile="buy4",
        account_id="acct-buy4",
        payload={
            "plan_type": "plus",
            "rate_limit_reached_type": "rate_limit_reached",
            "rate_limit": {
                "allowed": True,
                "limit_reached": True,
                "primary_window": {
                    "limit_window_seconds": 18000,
                    "used_percent": 40,
                    "reset_at": 1700003600,
                },
                "secondary_window": {
                    "limit_window_seconds": 604800,
                    "used_percent": 100,
                    "reset_at": 1700600000,
                },
            },
        },
    )

    error = build_codex_rate_limit_error(profile="buy4", result=result)

    assert error.type == "usage_limit_reached"
    assert error.error_extra_fields["plan_type"] == "plus"
    assert error.error_extra_fields["resets_at"] == 1700600000
    assert error.headers["x-codex-active-limit"] == "weekly-limit"
    assert error.headers["x-weekly-limit-primary-used-percent"] == "100"
    assert error.headers["x-weekly-limit-primary-reset-at"] == "1700600000"


def test_build_codex_rate_limit_error_for_free_profile_returns_usage_not_included() -> None:
    result = normalize_usage_payload(
        profile="free-profile",
        account_id="acct-free",
        payload={
            "plan_type": "free",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
            },
        },
    )

    error = build_codex_rate_limit_error(profile="free-profile", result=result)

    assert error.type == "usage_not_included"
    assert error.error_extra_fields["plan_type"] == "free"
