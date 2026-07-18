from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import litellm
from litellm._logging import verbose_router_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .authenticator import get_chatgpt_authenticator
from .common_utils import (
    ChatGPTAuthError,
    get_chatgpt_originator,
    get_chatgpt_user_agent,
)

DEFAULT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
DEFAULT_RATE_LIMIT_RESET_CREDITS_URL = (
    "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
)
DEFAULT_ACCOUNTS_CHECK_URL = (
    "https://chatgpt.com/backend-api/accounts/check/v4-2023-04-27"
)


@dataclass
class UsageWindow:
    label: str
    used_percent: float
    reset_at: Optional[int]
    limit_seconds: Optional[int] = None


@dataclass
class UsageResult:
    profile: str
    account_id: str
    plan: str
    credits_balance: Optional[float]
    windows: List[UsageWindow]
    status: str
    error: Optional[str] = None
    account_type: str = "unknown"
    rate_limit_allowed: Optional[bool] = None
    rate_limit_reached: Optional[bool] = None
    rate_limit_reached_type: Optional[str] = None
    has_active_subscription: Optional[bool] = None
    subscription_expires_at: Optional[int] = None
    subscription_renews_at: Optional[int] = None
    effective_available: Optional[bool] = None


@dataclass
class ChatGPTUsageSnapshot:
    profile: str
    result: UsageResult
    refreshed_at: float


@dataclass(frozen=True)
class ChatGPTPacingInfo:
    window_label: str
    window_limit_seconds: Optional[int]
    remaining_ratio: float
    effective_deadline_at: Optional[int]
    time_ratio: Optional[float]
    pace_ratio: Optional[float]


@dataclass
class RateLimitResetCreditsResult:
    profile: str
    account_id: str
    available_count: Optional[int]
    credits: List[Dict[str, Any]]
    status: str
    next_expires_at: Optional[int] = None
    error: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None


class ChatGPTUsageMetrics:
    def __init__(self, registry: Any) -> None:
        from prometheus_client import Gauge

        self.refresh_success = Gauge(
            "litellm_chatgpt_usage_refresh_success",
            "Whether the last usage refresh completed successfully.",
            registry=registry,
        )
        self.refresh_timestamp = Gauge(
            "litellm_chatgpt_usage_refresh_timestamp_seconds",
            "Unix timestamp of the last usage refresh attempt.",
            registry=registry,
        )
        self.profile_up = Gauge(
            "litellm_chatgpt_profile_up",
            "Whether usage data was fetched successfully for the profile.",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_available = Gauge(
            "litellm_chatgpt_profile_available",
            "Whether the profile can currently serve traffic (excludes free plans and rate-limit blocks).",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_plan_info = Gauge(
            "litellm_chatgpt_profile_plan_info",
            "Account type info for each profile (always 1 for the active account_type label).",
            labelnames=["profile", "account_type"],
            registry=registry,
        )
        self.profile_has_active_subscription = Gauge(
            "litellm_chatgpt_profile_has_active_subscription",
            "Whether the profile currently has an active subscription (from accounts/check when available).",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_subscription_expires_timestamp = Gauge(
            "litellm_chatgpt_profile_subscription_expires_timestamp_seconds",
            "Unix timestamp when the profile subscription expires (if available).",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_subscription_renews_timestamp = Gauge(
            "litellm_chatgpt_profile_subscription_renews_timestamp_seconds",
            "Unix timestamp when the profile subscription renews (if available).",
            labelnames=["profile"],
            registry=registry,
        )
        self.credits_balance = Gauge(
            "litellm_chatgpt_profile_credits_balance",
            "Current ChatGPT credits balance for the profile.",
            labelnames=["profile"],
            registry=registry,
        )
        self.window_used_percent = Gauge(
            "litellm_chatgpt_usage_window_used_percent",
            "Used percentage for a ChatGPT usage window.",
            labelnames=["profile", "window"],
            registry=registry,
        )
        self.window_used_ratio = Gauge(
            "litellm_chatgpt_usage_window_used_ratio",
            "Used ratio for a ChatGPT usage window.",
            labelnames=["profile", "window"],
            registry=registry,
        )
        self.window_limit_seconds = Gauge(
            "litellm_chatgpt_usage_window_limit_seconds",
            "Configured duration of the ChatGPT usage window in seconds.",
            labelnames=["profile", "window"],
            registry=registry,
        )
        self.window_reset_timestamp = Gauge(
            "litellm_chatgpt_usage_window_reset_timestamp_seconds",
            "Unix timestamp when the ChatGPT usage window resets.",
            labelnames=["profile", "window"],
            registry=registry,
        )
        self.window_remaining_seconds = Gauge(
            "litellm_chatgpt_usage_window_remaining_seconds",
            "Seconds remaining before the ChatGPT usage window resets.",
            labelnames=["profile", "window"],
            registry=registry,
        )
        self.profile_weekly_remaining_ratio = Gauge(
            "litellm_chatgpt_profile_weekly_remaining_ratio",
            "Remaining ratio for the preferred weekly pacing window.",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_effective_deadline_timestamp = Gauge(
            "litellm_chatgpt_profile_effective_deadline_timestamp_seconds",
            "Earliest routing deadline derived from weekly reset and subscription expiry.",
            labelnames=["profile"],
            registry=registry,
        )
        self._known_profiles: set[str] = set()
        self._known_windows: set[tuple[str, str]] = set()
        self._known_plan_labels: set[tuple[str, str]] = set()

    def update(
        self, results: List[UsageResult], refreshed_at: Optional[float] = None
    ) -> None:
        refreshed_at = refreshed_at or time.time()
        self.refresh_timestamp.set(refreshed_at)
        self.refresh_success.set(1)

        active_profiles: set[str] = set()
        active_windows: set[tuple[str, str]] = set()
        active_plan_labels: set[tuple[str, str]] = set()

        for result in results:
            profile = result.profile
            active_profiles.add(profile)
            is_ok = result.status == "ok"
            self.profile_up.labels(profile=profile).set(1 if is_ok else 0)

            effective_available = _effective_available_from_result(result)
            self.profile_available.labels(profile=profile).set(
                1 if effective_available else 0
            )

            account_type = _sanitize_account_type(result.account_type or result.plan)
            plan_label_key = (profile, account_type)
            active_plan_labels.add(plan_label_key)
            self.profile_plan_info.labels(
                profile=profile, account_type=account_type
            ).set(1)

            self.profile_has_active_subscription.labels(profile=profile).set(
                _optional_bool_to_gauge_value(result.has_active_subscription)
            )
            self.profile_subscription_expires_timestamp.labels(profile=profile).set(
                result.subscription_expires_at
                if result.subscription_expires_at is not None
                else float("nan")
            )
            self.profile_subscription_renews_timestamp.labels(profile=profile).set(
                result.subscription_renews_at
                if result.subscription_renews_at is not None
                else float("nan")
            )
            self.credits_balance.labels(profile=profile).set(
                result.credits_balance
                if result.credits_balance is not None
                else float("nan")
            )
            pacing_info = compute_chatgpt_pacing_info(result, now=refreshed_at)
            self.profile_weekly_remaining_ratio.labels(profile=profile).set(
                pacing_info.remaining_ratio if pacing_info is not None else float("nan")
            )
            self.profile_effective_deadline_timestamp.labels(profile=profile).set(
                float(pacing_info.effective_deadline_at)
                if pacing_info is not None
                and pacing_info.effective_deadline_at is not None
                else float("nan")
            )

            for window in result.windows:
                window_key = (profile, window.label)
                active_windows.add(window_key)
                remaining_seconds = (
                    max(float(window.reset_at) - refreshed_at, 0.0)
                    if window.reset_at is not None
                    else float("nan")
                )
                self.window_used_percent.labels(
                    profile=profile, window=window.label
                ).set(window.used_percent)
                self.window_used_ratio.labels(profile=profile, window=window.label).set(
                    window.used_percent / 100.0
                )
                self.window_limit_seconds.labels(
                    profile=profile, window=window.label
                ).set(
                    window.limit_seconds
                    if window.limit_seconds is not None
                    else float("nan")
                )
                self.window_reset_timestamp.labels(
                    profile=profile, window=window.label
                ).set(
                    float(window.reset_at)
                    if window.reset_at is not None
                    else float("nan")
                )
                self.window_remaining_seconds.labels(
                    profile=profile, window=window.label
                ).set(remaining_seconds)

        for profile in self._known_profiles - active_profiles:
            self.profile_up.remove(profile)
            self.profile_available.remove(profile)
            self.profile_has_active_subscription.remove(profile)
            self.profile_subscription_expires_timestamp.remove(profile)
            self.profile_subscription_renews_timestamp.remove(profile)
            self.credits_balance.remove(profile)
            self.profile_weekly_remaining_ratio.remove(profile)
            self.profile_effective_deadline_timestamp.remove(profile)

        for profile, window in self._known_windows - active_windows:
            self.window_used_percent.remove(profile, window)
            self.window_used_ratio.remove(profile, window)
            self.window_limit_seconds.remove(profile, window)
            self.window_reset_timestamp.remove(profile, window)
            self.window_remaining_seconds.remove(profile, window)

        for profile, account_type in self._known_plan_labels - active_plan_labels:
            self.profile_plan_info.remove(profile, account_type)

        self._known_profiles = active_profiles
        self._known_windows = active_windows
        self._known_plan_labels = active_plan_labels

    def mark_refresh_failure(self, refreshed_at: Optional[float] = None) -> None:
        self.refresh_timestamp.set(refreshed_at or time.time())
        self.refresh_success.set(0)


class ChatGPTRateLimitResetCreditsMetrics:
    def __init__(self, registry: Any) -> None:
        from prometheus_client import Gauge

        self.refresh_success = Gauge(
            "litellm_chatgpt_rate_limit_reset_refresh_success",
            "Whether the last rate-limit reset credits refresh completed successfully.",
            registry=registry,
        )
        self.refresh_timestamp = Gauge(
            "litellm_chatgpt_rate_limit_reset_refresh_timestamp_seconds",
            "Unix timestamp of the last rate-limit reset credits refresh attempt.",
            registry=registry,
        )
        self.profile_up = Gauge(
            "litellm_chatgpt_rate_limit_reset_profile_up",
            "Whether reset credit data was fetched successfully for the profile.",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_available_count = Gauge(
            "litellm_chatgpt_rate_limit_reset_available_count",
            "Number of ChatGPT rate-limit reset credits available for the profile.",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_credit_count = Gauge(
            "litellm_chatgpt_rate_limit_reset_credit_count",
            "Number of reset credit entries returned for the profile.",
            labelnames=["profile"],
            registry=registry,
        )
        self.profile_next_expires_timestamp = Gauge(
            "litellm_chatgpt_rate_limit_reset_next_expires_timestamp_seconds",
            "Unix timestamp when the earliest available reset credit expires.",
            labelnames=["profile"],
            registry=registry,
        )
        self.credit_expires_timestamp = Gauge(
            "litellm_chatgpt_rate_limit_reset_credit_expires_timestamp_seconds",
            "Unix timestamp when a reset credit expires.",
            labelnames=["profile", "credit_id", "status"],
            registry=registry,
        )
        self._known_profiles: set[str] = set()
        self._known_credit_labels: set[tuple[str, str, str]] = set()

    def update(
        self,
        results: List[RateLimitResetCreditsResult],
        refreshed_at: Optional[float] = None,
    ) -> None:
        refreshed_at = refreshed_at or time.time()
        self.refresh_timestamp.set(refreshed_at)
        self.refresh_success.set(1)

        active_profiles: set[str] = set()
        active_credit_labels: set[tuple[str, str, str]] = set()
        for result in results:
            profile = result.profile
            active_profiles.add(profile)
            self.profile_up.labels(profile=profile).set(
                1 if result.status == "ok" else 0
            )
            self.profile_available_count.labels(profile=profile).set(
                result.available_count
                if result.available_count is not None
                else float("nan")
            )
            self.profile_credit_count.labels(profile=profile).set(len(result.credits))
            self.profile_next_expires_timestamp.labels(profile=profile).set(
                float(result.next_expires_at)
                if result.next_expires_at is not None
                else float("nan")
            )

            for credit in result.credits:
                credit_id = get_rate_limit_reset_credit_id(credit)
                if credit_id is None:
                    continue
                status = _sanitize_reset_credit_status(credit.get("status"))
                label_key = (profile, credit_id, status)
                active_credit_labels.add(label_key)
                expires_at = get_rate_limit_reset_credit_expires_at(credit)
                self.credit_expires_timestamp.labels(
                    profile=profile,
                    credit_id=credit_id,
                    status=status,
                ).set(float(expires_at) if expires_at is not None else float("nan"))

        for profile in self._known_profiles - active_profiles:
            self.profile_up.remove(profile)
            self.profile_available_count.remove(profile)
            self.profile_credit_count.remove(profile)
            self.profile_next_expires_timestamp.remove(profile)

        for profile, credit_id, status in (
            self._known_credit_labels - active_credit_labels
        ):
            self.credit_expires_timestamp.remove(profile, credit_id, status)

        self._known_profiles = active_profiles
        self._known_credit_labels = active_credit_labels

    def mark_refresh_failure(self, refreshed_at: Optional[float] = None) -> None:
        self.refresh_timestamp.set(refreshed_at or time.time())
        self.refresh_success.set(0)


class ChatGPTCodexRateLimitError(BaseLLMException):
    def __init__(
        self,
        *,
        message: str,
        error_type: str,
        headers: Optional[Dict[str, str]] = None,
        error_extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(status_code=429, message=message, headers=headers)
        self.type = error_type
        self.param = None
        self.error_extra_fields = error_extra_fields or {}


_CODEX_RATE_LIMIT_REACHED_TYPES = {
    "rate_limit_reached",
    "workspace_owner_credits_depleted",
    "workspace_member_credits_depleted",
    "workspace_owner_usage_limit_reached",
    "workspace_member_usage_limit_reached",
}


def get_usage_url() -> str:
    return str(
        litellm.get_secret("LITELLM_CHATGPT_USAGE_URL")
        or litellm.get_secret("CHATGPT_USAGE_URL")
        or DEFAULT_USAGE_URL
    )


def get_rate_limit_reset_credits_url() -> str:
    return str(
        litellm.get_secret("LITELLM_CHATGPT_RATE_LIMIT_RESET_CREDITS_URL")
        or litellm.get_secret("CHATGPT_RATE_LIMIT_RESET_CREDITS_URL")
        or DEFAULT_RATE_LIMIT_RESET_CREDITS_URL
    )


def get_accounts_check_url() -> str:
    return str(
        litellm.get_secret("LITELLM_CHATGPT_ACCOUNTS_CHECK_URL")
        or litellm.get_secret("CHATGPT_ACCOUNTS_CHECK_URL")
        or DEFAULT_ACCOUNTS_CHECK_URL
    )


def _parse_optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _parse_optional_timestamp(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            return None
        if raw_value.isdigit():
            return int(raw_value)
        normalized = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
        try:
            return int(datetime.fromisoformat(normalized).timestamp())
        except ValueError:
            return None
    return None


def _sanitize_account_type(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower()
    return normalized if normalized else "unknown"


def _sanitize_reset_credit_status(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip().lower()
    return normalized if normalized else "unknown"


def _normalize_codex_rate_limit_reached_type(value: Optional[str]) -> str:
    normalized = _sanitize_account_type(value)
    if normalized in _CODEX_RATE_LIMIT_REACHED_TYPES:
        return normalized
    return "rate_limit_reached"


def _optional_bool_to_gauge_value(value: Optional[bool]) -> float:
    if value is None:
        return float("nan")
    return 1.0 if value else 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _compute_effective_available(
    *,
    status: str,
    account_type: str,
    rate_limit_allowed: Optional[bool],
    rate_limit_reached: Optional[bool],
    has_active_subscription: Optional[bool],
) -> bool:
    if status != "ok":
        return False

    normalized_account_type = _sanitize_account_type(account_type)
    if normalized_account_type == "free":
        return False

    if has_active_subscription is False:
        return False

    if rate_limit_allowed is False:
        return False

    if rate_limit_reached is True:
        return False

    return True


def _effective_available_from_result(result: UsageResult) -> bool:
    if result.effective_available is not None:
        return bool(result.effective_available)
    return _compute_effective_available(
        status=result.status,
        account_type=result.account_type,
        rate_limit_allowed=result.rate_limit_allowed,
        rate_limit_reached=result.rate_limit_reached,
        has_active_subscription=result.has_active_subscription,
    )


def get_rate_limit_reset_credit_id(credit: Dict[str, Any]) -> Optional[str]:
    for key in ("credit_id", "id"):
        value = credit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_rate_limit_reset_credit_expires_at(credit: Dict[str, Any]) -> Optional[int]:
    for key in (
        "expires_at_timestamp",
        "expires_at",
        "expire_at",
        "expiresAt",
        "expiration_time",
    ):
        expires_at = _parse_optional_timestamp(credit.get(key))
        if expires_at is not None:
            return expires_at
    return None


def is_rate_limit_reset_credit_available(credit: Dict[str, Any]) -> bool:
    if credit.get("available") is False:
        return False
    status = _sanitize_reset_credit_status(credit.get("status"))
    if status not in ("unknown", "available"):
        return False
    return get_rate_limit_reset_credit_id(credit) is not None


def get_next_rate_limit_reset_credit_expires_at(
    credits: Sequence[Dict[str, Any]],
) -> Optional[int]:
    expires_at_values = [
        expires_at
        for credit in credits
        if is_rate_limit_reset_credit_available(credit)
        for expires_at in [get_rate_limit_reset_credit_expires_at(credit)]
        if expires_at is not None
    ]
    if not expires_at_values:
        return None
    return min(expires_at_values)


def select_rate_limit_reset_credit_id(
    credits: Sequence[Dict[str, Any]],
) -> Optional[str]:
    candidates = [
        credit
        for credit in credits
        if isinstance(credit, dict) and is_rate_limit_reset_credit_available(credit)
    ]
    if not candidates:
        return None

    def _sort_key(credit: Dict[str, Any]) -> Tuple[int, int, str]:
        expires_at = get_rate_limit_reset_credit_expires_at(credit)
        credit_id = get_rate_limit_reset_credit_id(credit) or ""
        return (0 if expires_at is not None else 1, expires_at or 0, credit_id)

    selected = min(candidates, key=_sort_key)
    return get_rate_limit_reset_credit_id(selected)


def is_usage_result_expired(
    result: UsageResult, *, now: Optional[float] = None
) -> bool:
    if result.subscription_expires_at is None:
        return False
    reference_time = now if now is not None else time.time()
    return float(result.subscription_expires_at) <= reference_time


def get_weekly_pacing_window(result: UsageResult) -> Optional[UsageWindow]:
    preferred_window: Optional[UsageWindow] = None
    for window in result.windows:
        if window.label == "1w" or window.limit_seconds == 604800:
            preferred_window = window
            break

    if preferred_window is not None:
        return preferred_window

    windows_with_limits = [
        window
        for window in result.windows
        if isinstance(window.limit_seconds, int) and window.limit_seconds > 0
    ]
    if not windows_with_limits:
        return None

    return max(windows_with_limits, key=lambda window: int(window.limit_seconds or 0))


def get_effective_deadline_at(
    result: UsageResult, weekly_window: Optional[UsageWindow]
) -> Optional[int]:
    deadline_candidates: List[int] = []
    if weekly_window is not None and weekly_window.reset_at is not None:
        deadline_candidates.append(int(weekly_window.reset_at))
    if result.subscription_expires_at is not None:
        deadline_candidates.append(int(result.subscription_expires_at))
    if not deadline_candidates:
        return None
    return min(deadline_candidates)


def _get_codex_limit_metadata(
    result: UsageResult,
) -> Tuple[str, str, Optional[UsageWindow]]:
    weekly_window = get_weekly_pacing_window(result)
    if weekly_window is None:
        return "codex", "codex", None

    if weekly_window.label == "1w" or weekly_window.limit_seconds == 604800:
        return "weekly-limit", "weekly", weekly_window
    if weekly_window.label == "5h" or weekly_window.limit_seconds == 18000:
        return "five-hour-limit", "5-hour", weekly_window
    return "codex", weekly_window.label or "codex", weekly_window


def build_codex_rate_limit_headers(result: UsageResult) -> Dict[str, str]:
    active_limit, limit_name, primary_window = _get_codex_limit_metadata(result)
    headers: Dict[str, str] = {
        "x-codex-active-limit": active_limit,
        "x-codex-rate-limit-reached-type": _normalize_codex_rate_limit_reached_type(
            result.rate_limit_reached_type
        ),
    }

    if limit_name:
        headers[f"x-{active_limit}-limit-name"] = str(limit_name)

    if primary_window is not None:
        headers[f"x-{active_limit}-primary-used-percent"] = str(
            int(round(primary_window.used_percent))
        )
        if primary_window.limit_seconds is not None:
            headers[f"x-{active_limit}-primary-window-minutes"] = str(
                int(primary_window.limit_seconds // 60)
            )
        if primary_window.reset_at is not None:
            headers[f"x-{active_limit}-primary-reset-at"] = str(primary_window.reset_at)

    if result.credits_balance is not None:
        headers["x-codex-credits-balance"] = str(result.credits_balance)

    return headers


def build_codex_rate_limit_error(
    profile: str,
    result: UsageResult,
) -> ChatGPTCodexRateLimitError:
    _, _, weekly_window = _get_codex_limit_metadata(result)
    plan_type = _sanitize_account_type(result.account_type or result.plan)
    resets_at = (
        weekly_window.reset_at
        if weekly_window is not None and weekly_window.reset_at is not None
        else get_effective_deadline_at(result, weekly_window)
    )

    if plan_type == "free" or result.has_active_subscription is False:
        return ChatGPTCodexRateLimitError(
            message=f"ChatGPT usage is not included for profile '{profile}'.",
            error_type="usage_not_included",
            error_extra_fields={"plan_type": plan_type},
        )

    error_extra_fields: Dict[str, Any] = {"plan_type": plan_type}
    if resets_at is not None:
        error_extra_fields["resets_at"] = int(resets_at)

    return ChatGPTCodexRateLimitError(
        message=f"ChatGPT usage limit reached for profile '{profile}'.",
        error_type="usage_limit_reached",
        headers=build_codex_rate_limit_headers(result),
        error_extra_fields=error_extra_fields,
    )


def compute_chatgpt_pacing_info(
    result: UsageResult,
    *,
    now: Optional[float] = None,
    min_time_ratio: float = 0.02,
) -> Optional[ChatGPTPacingInfo]:
    weekly_window = get_weekly_pacing_window(result)
    if weekly_window is None:
        return None

    remaining_ratio = _clamp(1.0 - (weekly_window.used_percent / 100.0), 0.0, 1.0)
    effective_deadline_at = get_effective_deadline_at(result, weekly_window)
    window_limit_seconds = weekly_window.limit_seconds

    if (
        effective_deadline_at is None
        or window_limit_seconds is None
        or window_limit_seconds <= 0
    ):
        return ChatGPTPacingInfo(
            window_label=weekly_window.label,
            window_limit_seconds=window_limit_seconds,
            remaining_ratio=remaining_ratio,
            effective_deadline_at=effective_deadline_at,
            time_ratio=None,
            pace_ratio=None,
        )

    reference_time = now if now is not None else time.time()
    remaining_seconds = max(float(effective_deadline_at) - reference_time, 0.0)
    bounded_min_time_ratio = _clamp(float(min_time_ratio), 1e-9, 1.0)
    time_ratio = _clamp(
        remaining_seconds / float(window_limit_seconds),
        bounded_min_time_ratio,
        1.0,
    )
    pace_ratio = remaining_ratio / time_ratio
    return ChatGPTPacingInfo(
        window_label=weekly_window.label,
        window_limit_seconds=window_limit_seconds,
        remaining_ratio=remaining_ratio,
        effective_deadline_at=effective_deadline_at,
        time_ratio=time_ratio,
        pace_ratio=pace_ratio,
    )


def _extract_account_candidates(
    payload: Dict[str, Any], account_id: str
) -> List[Dict[str, Any]]:
    raw_accounts = payload.get("accounts")
    candidates: List[Dict[str, Any]] = []

    def _append_candidate(candidate: Any) -> None:
        if isinstance(candidate, dict):
            candidates.append(candidate)

    if isinstance(raw_accounts, dict):
        if account_id:
            _append_candidate(raw_accounts.get(account_id))
        _append_candidate(raw_accounts.get("default"))
        for candidate in raw_accounts.values():
            _append_candidate(candidate)
    elif isinstance(raw_accounts, list):
        for candidate in raw_accounts:
            _append_candidate(candidate)

    seen_ids: set[int] = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        deduped.append(candidate)
    return deduped


def _extract_account_metadata(
    payload: Dict[str, Any], account_id: str
) -> Tuple[Optional[str], Optional[bool], Optional[int], Optional[int]]:
    account_type: Optional[str] = None
    has_active_subscription: Optional[bool] = None
    subscription_expires_at: Optional[int] = None
    subscription_renews_at: Optional[int] = None

    for candidate in _extract_account_candidates(payload, account_id):
        account_block = candidate.get("account")
        entitlement_block = candidate.get("entitlement")

        candidate_account_type = None
        if isinstance(account_block, dict):
            candidate_account_type = account_block.get("plan_type")
        if candidate_account_type is None:
            candidate_account_type = candidate.get("plan_type")

        if account_type is None and candidate_account_type is not None:
            account_type = _sanitize_account_type(candidate_account_type)

        if isinstance(entitlement_block, dict):
            if has_active_subscription is None and isinstance(
                entitlement_block.get("has_active_subscription"), bool
            ):
                has_active_subscription = entitlement_block.get(
                    "has_active_subscription"
                )

            if subscription_expires_at is None:
                subscription_expires_at = _parse_optional_timestamp(
                    entitlement_block.get("expires_at")
                )

            if subscription_renews_at is None:
                subscription_renews_at = _parse_optional_timestamp(
                    entitlement_block.get("renews_at")
                )

        if account_type is not None and (
            has_active_subscription is not None
            or subscription_expires_at is not None
            or subscription_renews_at is not None
        ):
            break

    return (
        account_type,
        has_active_subscription,
        subscription_expires_at,
        subscription_renews_at,
    )


def _format_window_label(limit_seconds: Optional[int], default_hours: int) -> str:
    if limit_seconds is None or limit_seconds <= 0:
        return f"{default_hours}h"
    if limit_seconds % 604800 == 0:
        weeks = limit_seconds // 604800
        return f"{weeks}w"
    if limit_seconds % 86400 == 0:
        days = limit_seconds // 86400
        return f"{days}d"
    if limit_seconds % 3600 == 0:
        hours = limit_seconds // 3600
        return f"{hours}h"
    if limit_seconds % 60 == 0:
        minutes = limit_seconds // 60
        return f"{minutes}m"
    return f"{limit_seconds}s"


def _load_auth_data(profile: str) -> Tuple[Any, Dict[str, Any]]:
    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_data = authenticator._read_auth_file() or {}
    return authenticator, auth_data


def _get_usage_access_token(authenticator: Any, auth_data: Dict[str, Any]) -> str:
    access_token = auth_data.get("access_token")
    if access_token and not authenticator._is_token_expired(auth_data, access_token):
        return access_token

    refresh_token = auth_data.get("refresh_token")
    if refresh_token:
        refreshed = authenticator._refresh_tokens(refresh_token)
        return refreshed["access_token"]

    raise ValueError(
        f"Profile '{authenticator.profile_name}' is not logged in. Run `litellm-chatgpt login {authenticator.profile_name}`."
    )


def _build_chatgpt_api_headers(
    access_token: str,
    account_id: str,
    *,
    content_type_json: bool = False,
) -> Dict[str, str]:
    originator = get_chatgpt_originator()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": get_chatgpt_user_agent(originator),
        "originator": originator,
        "Accept": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    if content_type_json:
        headers["Content-Type"] = "application/json"
    return headers


def _normalize_usage_window(label: str, window_payload: Dict[str, Any]) -> UsageWindow:
    reset_at = _parse_optional_int(window_payload.get("reset_at"))
    limit_seconds = _parse_optional_int(window_payload.get("limit_window_seconds"))

    used_percent_raw = window_payload.get("used_percent", 0)
    try:
        used_percent = float(used_percent_raw)
    except (TypeError, ValueError):
        used_percent = 0.0

    return UsageWindow(
        label=label,
        used_percent=max(0.0, min(100.0, used_percent)),
        reset_at=reset_at,
        limit_seconds=limit_seconds,
    )


def normalize_usage_payload(
    profile: str, account_id: str, payload: Dict[str, Any]
) -> UsageResult:
    plan = _sanitize_account_type(payload.get("plan_type"))
    credits_balance = None
    credits = payload.get("credits")
    if isinstance(credits, dict) and credits.get("balance") is not None:
        try:
            credits_balance = float(credits["balance"])
        except (TypeError, ValueError):
            credits_balance = None

    rate_limit_allowed: Optional[bool] = None
    rate_limit_reached: Optional[bool] = None

    windows: List[UsageWindow] = []
    rate_limit = payload.get("rate_limit")
    if isinstance(rate_limit, dict):
        if isinstance(rate_limit.get("allowed"), bool):
            rate_limit_allowed = rate_limit.get("allowed")
        if isinstance(rate_limit.get("limit_reached"), bool):
            rate_limit_reached = rate_limit.get("limit_reached")

        primary = rate_limit.get("primary_window")
        if isinstance(primary, dict):
            windows.append(
                _normalize_usage_window(
                    _format_window_label(
                        _parse_optional_int(primary.get("limit_window_seconds")),
                        default_hours=3,
                    ),
                    primary,
                )
            )
        secondary = rate_limit.get("secondary_window")
        if isinstance(secondary, dict):
            windows.append(
                _normalize_usage_window(
                    _format_window_label(
                        _parse_optional_int(secondary.get("limit_window_seconds")),
                        default_hours=24,
                    ),
                    secondary,
                )
            )

    rate_limit_reached_type = None
    rate_limit_reached_raw = payload.get("rate_limit_reached_type")
    if isinstance(rate_limit_reached_raw, dict):
        if rate_limit_reached_raw.get("type") is not None:
            rate_limit_reached_type = str(rate_limit_reached_raw.get("type"))
    elif isinstance(rate_limit_reached_raw, str):
        rate_limit_reached_type = rate_limit_reached_raw

    result = UsageResult(
        profile=profile,
        account_id=account_id,
        plan=plan,
        credits_balance=credits_balance,
        windows=windows,
        status="ok",
        account_type=plan,
        rate_limit_allowed=rate_limit_allowed,
        rate_limit_reached=rate_limit_reached,
        rate_limit_reached_type=rate_limit_reached_type,
    )
    result.effective_available = _compute_effective_available(
        status=result.status,
        account_type=result.account_type,
        rate_limit_allowed=result.rate_limit_allowed,
        rate_limit_reached=result.rate_limit_reached,
        has_active_subscription=result.has_active_subscription,
    )
    return result


def _normalize_rate_limit_reset_credit(credit: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(credit)
    expires_at = get_rate_limit_reset_credit_expires_at(normalized)
    if expires_at is not None:
        normalized["expires_at_timestamp"] = expires_at
    return normalized


def _normalize_rate_limit_reset_credits_payload(
    profile: str,
    account_id: str,
    payload: Dict[str, Any],
) -> RateLimitResetCreditsResult:
    available_count = _parse_optional_int(payload.get("available_count"))
    credits_payload = payload.get("credits")
    credits: List[Dict[str, Any]] = []
    if isinstance(credits_payload, list):
        credits = [
            _normalize_rate_limit_reset_credit(credit)
            for credit in credits_payload
            if isinstance(credit, dict)
        ]
    elif isinstance(credits_payload, dict):
        credits = [_normalize_rate_limit_reset_credit(credits_payload)]

    code = payload.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)

    message = payload.get("message")
    if message is None and isinstance(payload.get("detail"), str):
        message = payload.get("detail")
    if message is not None and not isinstance(message, str):
        message = str(message)

    status = "ok"
    error = None
    if code not in (None, "", "reset"):
        status = "error"
        error = message or f"reset credits returned code {code}"

    return RateLimitResetCreditsResult(
        profile=profile,
        account_id=account_id,
        available_count=available_count,
        credits=credits,
        status=status,
        next_expires_at=get_next_rate_limit_reset_credit_expires_at(credits),
        error=error,
        code=code,
        message=message,
        raw_payload=payload,
    )


def _fetch_account_metadata(
    client: Any,
    access_token: str,
    account_id: str,
    accounts_check_url: Optional[str] = None,
) -> Tuple[Optional[str], Optional[bool], Optional[int], Optional[int]]:
    response = client.get(
        accounts_check_url or get_accounts_check_url(),
        headers=_build_chatgpt_api_headers(access_token, account_id),
    )

    try:
        response.raise_for_status()
    except Exception as exc:
        verbose_router_logger.debug(
            "ChatGPTUsageService: accounts/check request failed profile_account_id=%s status=%s error=%s",
            account_id,
            getattr(response, "status_code", "unknown"),
            exc,
        )
        return (None, None, None, None)

    try:
        payload = response.json()
    except Exception as exc:
        verbose_router_logger.debug(
            "ChatGPTUsageService: accounts/check json parse failed profile_account_id=%s error=%s",
            account_id,
            exc,
        )
        return (None, None, None, None)

    if not isinstance(payload, dict):
        return (None, None, None, None)

    return _extract_account_metadata(payload, account_id)


def fetch_usage_for_profile(
    profile: str, usage_url: Optional[str] = None
) -> UsageResult:
    account_id = ""
    try:
        authenticator, auth_data = _load_auth_data(profile)
        account_id = auth_data.get("account_id") or authenticator.get_account_id() or ""
        access_token = _get_usage_access_token(authenticator, auth_data)
    except (ChatGPTAuthError, ValueError) as exc:
        return UsageResult(
            profile=profile,
            account_id=account_id,
            plan="N/A",
            credits_balance=None,
            windows=[],
            status="error",
            error=f"usage auth failed: {exc}",
            effective_available=False,
        )

    client = _get_httpx_client()
    response = client.get(
        usage_url or get_usage_url(),
        headers=_build_chatgpt_api_headers(access_token, account_id),
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        body_text = response.text.strip() if response.text else str(exc)
        return UsageResult(
            profile=profile,
            account_id=account_id,
            plan="N/A",
            credits_balance=None,
            windows=[],
            status="error",
            error=f"usage request failed ({response.status_code}): {body_text}",
            effective_available=False,
        )

    payload = response.json()
    result = normalize_usage_payload(profile, account_id, payload)

    (
        account_type,
        has_active_subscription,
        subscription_expires_at,
        subscription_renews_at,
    ) = _fetch_account_metadata(client, access_token, account_id)

    if account_type is not None:
        result.account_type = account_type
    result.has_active_subscription = has_active_subscription
    result.subscription_expires_at = subscription_expires_at
    result.subscription_renews_at = subscription_renews_at
    result.effective_available = _compute_effective_available(
        status=result.status,
        account_type=result.account_type,
        rate_limit_allowed=result.rate_limit_allowed,
        rate_limit_reached=result.rate_limit_reached,
        has_active_subscription=result.has_active_subscription,
    )
    return result


def fetch_rate_limit_reset_credits_for_profile(
    profile: str,
    reset_credits_url: Optional[str] = None,
) -> RateLimitResetCreditsResult:
    account_id = ""
    try:
        authenticator, auth_data = _load_auth_data(profile)
        account_id = auth_data.get("account_id") or authenticator.get_account_id() or ""
        access_token = _get_usage_access_token(authenticator, auth_data)
    except (ChatGPTAuthError, ValueError) as exc:
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits auth failed: {exc}",
        )

    client = _get_httpx_client()
    response = client.get(
        reset_credits_url or get_rate_limit_reset_credits_url(),
        headers=_build_chatgpt_api_headers(access_token, account_id),
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        body_text = response.text.strip() if response.text else str(exc)
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits request failed ({response.status_code}): {body_text}",
        )

    try:
        payload = response.json()
    except Exception as exc:
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits response parse failed: {exc}",
        )

    if not isinstance(payload, dict):
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits response must be a JSON object: {payload!r}",
        )

    return _normalize_rate_limit_reset_credits_payload(profile, account_id, payload)


def consume_rate_limit_reset_credit_for_profile(
    profile: str,
    credit_id: str,
    redeem_request_id: str,
    reset_credits_url: Optional[str] = None,
) -> RateLimitResetCreditsResult:
    account_id = ""
    try:
        authenticator, auth_data = _load_auth_data(profile)
        account_id = auth_data.get("account_id") or authenticator.get_account_id() or ""
        access_token = _get_usage_access_token(authenticator, auth_data)
    except (ChatGPTAuthError, ValueError) as exc:
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits auth failed: {exc}",
        )

    client = _get_httpx_client()
    endpoint = (reset_credits_url or get_rate_limit_reset_credits_url()).rstrip(
        "/"
    ) + "/consume"
    response = client.post(
        endpoint,
        headers=_build_chatgpt_api_headers(
            access_token, account_id, content_type_json=True
        ),
        json={"credit_id": credit_id, "redeem_request_id": redeem_request_id},
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        body_text = response.text.strip() if response.text else str(exc)
        return RateLimitResetCreditsResult(
            profile=profile,
            account_id=account_id,
            available_count=None,
            credits=[],
            status="error",
            error=f"reset credits consume failed ({response.status_code}): {body_text}",
        )

    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text.strip() if response.text else ""}

    if not isinstance(payload, dict):
        payload = {"raw_payload": payload}

    return _normalize_rate_limit_reset_credits_payload(profile, account_id, payload)


class ChatGPTUsageService:
    def __init__(
        self,
        refresh_interval_seconds: int = 60,
        stale_after_seconds: int = 180,
        usage_url: Optional[str] = None,
    ) -> None:
        self.refresh_interval_seconds = max(1, refresh_interval_seconds)
        self.stale_after_seconds = max(1, stale_after_seconds)
        self.usage_url = usage_url or get_usage_url()
        self._snapshots: Dict[str, ChatGPTUsageSnapshot] = {}
        self._profile_locks: Dict[str, asyncio.Lock] = {}
        self._profile_sync_locks: Dict[str, threading.Lock] = {}
        self._known_profiles: set[str] = set()
        self._background_refresh_task: Optional[asyncio.Task[Any]] = None

    @staticmethod
    def get_profile_name_from_deployment(deployment: dict) -> Optional[str]:
        litellm_params = deployment.get("litellm_params")
        if not isinstance(litellm_params, dict):
            return None
        profile = litellm_params.get("chatgpt_auth_profile")
        if profile is None:
            return None
        return str(profile)

    def register_deployments(self, deployments: Sequence[dict]) -> None:
        for deployment in deployments:
            profile = self.get_profile_name_from_deployment(deployment)
            if profile:
                self._known_profiles.add(profile)

    def ensure_background_refresh_task(self) -> None:
        if not self._known_profiles:
            return
        if (
            self._background_refresh_task is not None
            and not self._background_refresh_task.done()
        ):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._background_refresh_task = loop.create_task(
            self._background_refresh_loop()
        )

    async def get_snapshot(
        self, profile: str, *, allow_stale: bool = True
    ) -> Optional[ChatGPTUsageSnapshot]:
        self._known_profiles.add(profile)
        self.ensure_background_refresh_task()

        snapshot = self._snapshots.get(profile)
        if snapshot is not None and not self._is_stale(snapshot):
            return snapshot

        await self.refresh_profile(profile)
        snapshot = self._snapshots.get(profile)
        if snapshot is None:
            return None
        if allow_stale or not self._is_stale(snapshot):
            return snapshot
        return None

    def get_snapshot_sync(
        self, profile: str, *, allow_stale: bool = True
    ) -> Optional[ChatGPTUsageSnapshot]:
        self._known_profiles.add(profile)

        snapshot = self._snapshots.get(profile)
        if snapshot is not None and not self._is_stale(snapshot):
            return snapshot

        snapshot = self.refresh_profile_sync(profile)
        if snapshot is None:
            return None
        if allow_stale or not self._is_stale(snapshot):
            return snapshot
        return None

    async def refresh_profile(self, profile: str) -> Optional[ChatGPTUsageSnapshot]:
        lock = self._profile_locks.setdefault(profile, asyncio.Lock())
        async with lock:
            snapshot = self._snapshots.get(profile)
            if snapshot is not None and not self._is_stale(snapshot):
                return snapshot

            result = await asyncio.to_thread(
                fetch_usage_for_profile,
                profile,
                self.usage_url,
            )
            snapshot = ChatGPTUsageSnapshot(
                profile=profile,
                result=result,
                refreshed_at=time.time(),
            )
            self._snapshots[profile] = snapshot
            return snapshot

    def refresh_profile_sync(self, profile: str) -> Optional[ChatGPTUsageSnapshot]:
        lock = self._profile_sync_locks.setdefault(profile, threading.Lock())
        with lock:
            snapshot = self._snapshots.get(profile)
            if snapshot is not None and not self._is_stale(snapshot):
                return snapshot

            result = fetch_usage_for_profile(profile, self.usage_url)
            snapshot = ChatGPTUsageSnapshot(
                profile=profile,
                result=result,
                refreshed_at=time.time(),
            )
            self._snapshots[profile] = snapshot
            return snapshot

    async def _background_refresh_loop(self) -> None:
        while True:
            try:
                profiles = sorted(self._known_profiles)
                for profile in profiles:
                    try:
                        await self.refresh_profile(profile)
                    except Exception as exc:
                        verbose_router_logger.debug(
                            "ChatGPTUsageService: failed refreshing usage for profile=%s error=%s",
                            profile,
                            exc,
                        )
                await asyncio.sleep(self.refresh_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                verbose_router_logger.debug(
                    "ChatGPTUsageService: background refresh loop error=%s", exc
                )
                await asyncio.sleep(self.refresh_interval_seconds)

    def _is_stale(self, snapshot: ChatGPTUsageSnapshot) -> bool:
        return (time.time() - snapshot.refreshed_at) > self.stale_after_seconds
