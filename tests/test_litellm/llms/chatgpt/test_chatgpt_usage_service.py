from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.common_utils import RefreshAccessTokenError
from litellm.llms.chatgpt.usage_service import (
    _extract_account_metadata,
    fetch_usage_for_profile,
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
