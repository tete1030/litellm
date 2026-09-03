"""HTTP application-body traffic accounting for the LiteLLM proxy.

The counters in this module intentionally measure HTTP bodies after LiteLLM's
request and response transformations. They do not claim to be wire bytes:
HTTP headers, transfer framing, and TLS are only visible to the edge proxy.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

TRAFFIC_IDENTITY_SCOPE_KEY = "litellm_traffic_identity"
_daily_aggregate_error_reported = False


def set_request_traffic_identity(
    *,
    request: Any,
    hashed_api_key: Optional[str],
    api_key_alias: Optional[str],
    route: Optional[str],
    requested_model: Optional[str],
    stream: Optional[bool],
) -> None:
    """Attach non-secret request dimensions for the outer ASGI middleware."""
    state = request.scope.setdefault("state", {})
    state[TRAFFIC_IDENTITY_SCOPE_KEY] = {
        "hashed_api_key": hashed_api_key,
        "api_key_alias": api_key_alias,
        "route": route,
        "requested_model": requested_model,
        "stream": stream,
    }


def _get_prometheus_logger() -> Optional[Any]:
    """Return the configured callback without creating a second registry."""
    try:
        from litellm.router_utils.cooldown_callbacks import (
            _get_prometheus_logger_from_callbacks,
        )

        return _get_prometheus_logger_from_callbacks()
    except Exception:
        return None


def _daily_aggregate_storage_enabled() -> bool:
    return os.getenv("LITELLM_TRAFFIC_METRICS_DAILY_DB_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _daily_aggregate_date() -> str:
    """Return the aggregate day in the explicitly configured reporting zone."""
    timezone_name = os.getenv("LITELLM_TRAFFIC_METRICS_DAILY_TIMEZONE", "UTC")
    try:
        return datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    except ZoneInfoNotFoundError:
        verbose_proxy_logger.warning(
            "Invalid LITELLM_TRAFFIC_METRICS_DAILY_TIMEZONE=%s; using UTC.",
            timezone_name,
        )
        return datetime.now(timezone.utc).date().isoformat()


def _schedule_daily_traffic_aggregate(
    *,
    hashed_api_key: Optional[str],
    api_key_alias: Optional[str],
    route: Optional[str],
    requested_model: Optional[str],
    client_request_body_bytes: int = 0,
    client_response_body_bytes: int = 0,
    provider_request_body_bytes: int = 0,
    provider_response_body_bytes: int = 0,
    client_requests: int = 0,
    provider_attempts: int = 0,
) -> None:
    """Persist only daily aggregate deltas; never enqueue request payloads."""
    if not _daily_aggregate_storage_enabled() or not hashed_api_key:
        return
    try:
        asyncio.get_running_loop().create_task(
            _upsert_daily_traffic_aggregate(
                hashed_api_key=hashed_api_key,
                api_key_alias=api_key_alias,
                route=route,
                requested_model=requested_model,
                client_request_body_bytes=client_request_body_bytes,
                client_response_body_bytes=client_response_body_bytes,
                provider_request_body_bytes=provider_request_body_bytes,
                provider_response_body_bytes=provider_response_body_bytes,
                client_requests=client_requests,
                provider_attempts=provider_attempts,
            )
        )
    except RuntimeError:
        return


async def _upsert_daily_traffic_aggregate(
    *,
    hashed_api_key: str,
    api_key_alias: Optional[str],
    route: Optional[str],
    requested_model: Optional[str],
    client_request_body_bytes: int,
    client_response_body_bytes: int,
    provider_request_body_bytes: int,
    provider_response_body_bytes: int,
    client_requests: int,
    provider_attempts: int,
) -> None:
    global _daily_aggregate_error_reported
    try:
        from litellm.proxy import proxy_server

        prisma_client = getattr(proxy_server, "prisma_client", None)
        table = getattr(
            getattr(prisma_client, "db", None), "litellm_dailytraffic", None
        )
        if table is None:
            return

        date = _daily_aggregate_date()
        api_key_alias = api_key_alias or ""
        route = route or ""
        requested_model = requested_model or ""
        create_data = {
            "date": date,
            "api_key": hashed_api_key,
            "api_key_alias": api_key_alias,
            "route": route,
            "requested_model": requested_model,
            "client_request_body_bytes": client_request_body_bytes,
            "client_response_body_bytes": client_response_body_bytes,
            "provider_request_body_bytes": provider_request_body_bytes,
            "provider_response_body_bytes": provider_response_body_bytes,
            "client_requests": client_requests,
            "provider_attempts": provider_attempts,
        }
        update_data: Dict[str, Any] = {
            field: {"increment": value}
            for field, value in create_data.items()
            if field.endswith("_bytes")
            or field in {"client_requests", "provider_attempts"}
            if value > 0
        }
        if api_key_alias:
            update_data["api_key_alias"] = api_key_alias
        await table.upsert(
            where={
                "date_api_key_route_requested_model": {
                    "date": date,
                    "api_key": hashed_api_key,
                    "route": route,
                    "requested_model": requested_model,
                }
            },
            data={"create": create_data, "update": update_data},
        )
    except Exception:
        # The Postgres aggregate is optional; a failed write must not affect API traffic.
        if not _daily_aggregate_error_reported:
            _daily_aggregate_error_reported = True
            verbose_proxy_logger.warning(
                "Daily traffic aggregate write failed; Prometheus traffic metrics remain available. "
                "Confirm the LiteLLM traffic migration was applied."
            )
        return


def _normalised_scope_route(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return str(scope.get("path") or "")


def _observe_client_body_bytes(
    scope: Scope,
    request_body_bytes: int,
    response_body_bytes: int,
    status_code: Optional[int],
) -> None:
    state = scope.get("state") or {}
    identity: Dict[str, Any] = state.get(TRAFFIC_IDENTITY_SCOPE_KEY) or {}
    route = identity.get("route") or _normalised_scope_route(scope)
    _schedule_daily_traffic_aggregate(
        hashed_api_key=identity.get("hashed_api_key"),
        api_key_alias=identity.get("api_key_alias"),
        route=route,
        requested_model=identity.get("requested_model"),
        client_request_body_bytes=request_body_bytes,
        client_response_body_bytes=response_body_bytes,
        client_requests=1,
    )
    prometheus_logger = _get_prometheus_logger()
    if prometheus_logger is None:
        return
    try:
        prometheus_logger.observe_client_body_bytes(
            hashed_api_key=identity.get("hashed_api_key"),
            api_key_alias=identity.get("api_key_alias"),
            requested_model=identity.get("requested_model"),
            route=route,
            status_code=status_code,
            stream=identity.get("stream"),
            request_body_bytes=request_body_bytes,
            response_body_bytes=response_body_bytes,
        )
    except Exception:
        # Accounting must never alter a client response.
        return


class TrafficMetricsMiddleware:
    """Measure request and response bodies at LiteLLM's client-facing boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "/metrics" in str(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        request_body_bytes = 0
        response_body_bytes = 0
        status_code: Optional[int] = None

        async def receive_with_count() -> Dict[str, Any]:
            nonlocal request_body_bytes
            message = await receive()
            if message["type"] == "http.request":
                request_body_bytes += len(message.get("body", b""))
            return message

        async def send_with_count(message: Dict[str, Any]) -> None:
            nonlocal response_body_bytes, status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            elif message["type"] == "http.response.body":
                response_body_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive_with_count, send_with_count)
        finally:
            _observe_client_body_bytes(
                scope,
                request_body_bytes=request_body_bytes,
                response_body_bytes=response_body_bytes,
                status_code=status_code,
            )


class ProviderTrafficObserver:
    """Defers provider response accounting until a streaming body is consumed."""

    def __init__(
        self,
        *,
        hashed_api_key: Optional[str],
        api_key_alias: Optional[str],
        requested_model: Optional[str],
        route: Optional[str],
        model: Optional[str],
        model_id: Optional[str],
        api_provider: str,
        chatgpt_auth_profile: Optional[str],
        attempt: int,
        stream: bool,
    ) -> None:
        self.hashed_api_key = hashed_api_key
        self.api_key_alias = api_key_alias
        self.requested_model = requested_model
        self.route = route
        self.model = model
        self.model_id = model_id
        self.api_provider = api_provider
        self.chatgpt_auth_profile = chatgpt_auth_profile
        self.attempt = attempt
        self.stream = stream
        self.request_body_bytes = 0
        self.response_body_bytes = 0
        self._finished = False

    def record_request_body(self, request_body: bytes) -> None:
        self.request_body_bytes += len(request_body)

    def wrap_async_response(
        self, response: httpx.Response, stream: bool
    ) -> httpx.Response:
        if stream and not response.is_stream_consumed:
            response.stream = _CountingAsyncByteStream(response.stream, self, response)
        else:
            self.response_body_bytes += len(response.content)
            self.finish(response.status_code)
        return response

    def wrap_sync_response(
        self, response: httpx.Response, stream: bool
    ) -> httpx.Response:
        if stream and not response.is_stream_consumed:
            response.stream = _CountingSyncByteStream(response.stream, self, response)
        else:
            self.response_body_bytes += len(response.content)
            self.finish(response.status_code)
        return response

    def add_response_body_bytes(self, chunk: bytes) -> None:
        self.response_body_bytes += len(chunk)

    def finish(self, status_code: Optional[int]) -> None:
        if self._finished:
            return
        self._finished = True
        _schedule_daily_traffic_aggregate(
            hashed_api_key=self.hashed_api_key,
            api_key_alias=self.api_key_alias,
            route=self.route,
            requested_model=self.requested_model,
            provider_request_body_bytes=self.request_body_bytes,
            provider_response_body_bytes=self.response_body_bytes,
            provider_attempts=1,
        )
        prometheus_logger = _get_prometheus_logger()
        if prometheus_logger is None:
            return
        try:
            prometheus_logger.observe_provider_body_bytes(
                hashed_api_key=self.hashed_api_key,
                api_key_alias=self.api_key_alias,
                requested_model=self.requested_model,
                route=self.route,
                model=self.model,
                model_id=self.model_id,
                api_provider=self.api_provider,
                chatgpt_auth_profile=self.chatgpt_auth_profile,
                status_code=status_code,
                attempt=self.attempt,
                stream=self.stream,
                request_body_bytes=self.request_body_bytes,
                response_body_bytes=self.response_body_bytes,
            )
        except Exception:
            return

    def record_error(self, error: Exception) -> None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None) or getattr(
            error, "status_code", None
        )
        self.finish(status_code)


class _CountingAsyncByteStream(httpx.AsyncByteStream):
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        observer: ProviderTrafficObserver,
        response: httpx.Response,
    ) -> None:
        self.stream = stream
        self.observer = observer
        self.response = response

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                self.observer.add_response_body_bytes(chunk)
                yield chunk
        finally:
            self.observer.finish(self.response.status_code)

    async def aclose(self) -> None:
        try:
            await self.stream.aclose()
        finally:
            self.observer.finish(self.response.status_code)


class _CountingSyncByteStream(httpx.SyncByteStream):
    def __init__(
        self,
        stream: httpx.SyncByteStream,
        observer: ProviderTrafficObserver,
        response: httpx.Response,
    ) -> None:
        self.stream = stream
        self.observer = observer
        self.response = response

    def __iter__(self):
        try:
            for chunk in self.stream:
                self.observer.add_response_body_bytes(chunk)
                yield chunk
        finally:
            self.observer.finish(self.response.status_code)

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.observer.finish(self.response.status_code)


def create_chatgpt_provider_traffic_observer(
    logging_obj: Optional[Any], stream: bool
) -> Optional[ProviderTrafficObserver]:
    """Build an observer only for the ChatGPT subscription provider."""
    model_call_details = getattr(logging_obj, "model_call_details", {}) or {}
    api_provider = str(model_call_details.get("custom_llm_provider") or "")
    if api_provider != "chatgpt":
        return None

    litellm_params = model_call_details.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    litellm_metadata = litellm_params.get("litellm_metadata") or {}
    request_metadata = {**litellm_metadata, **metadata}
    model_info = request_metadata.get("model_info") or {}
    raw_attempt = model_call_details.get("traffic_provider_attempt", 1)
    try:
        attempt = max(1, int(raw_attempt))
    except (TypeError, ValueError):
        attempt = 1

    return ProviderTrafficObserver(
        hashed_api_key=request_metadata.get("user_api_key_hash"),
        api_key_alias=request_metadata.get("user_api_key_alias"),
        requested_model=request_metadata.get("model_group")
        or litellm_params.get("model"),
        route=request_metadata.get("user_api_key_request_route"),
        model=model_call_details.get("model") or litellm_params.get("model"),
        model_id=model_info.get("id"),
        api_provider=api_provider,
        chatgpt_auth_profile=(
            request_metadata.get("chatgpt_auth_profile")
            or litellm_params.get("chatgpt_auth_profile")
        ),
        attempt=attempt,
        stream=stream,
    )
