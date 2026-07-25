from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import httpx
import litellm
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from .authenticator import Authenticator, get_chatgpt_authenticator
from .common_utils import ChatGPTAuthError, ChatGPTAuthProfileError
from .inventory_tools import load_inventory, render_config

DEFAULT_CALL_UPSTREAM = "https://chatgpt.com/backend-api/codex/realtime/calls"
DEFAULT_WS_UPSTREAM = "wss://api.openai.com/v1/live"
DEFAULT_PROFILE = "my"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4030
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0

CALL_PATH = "/backend-api/codex/realtime/calls"
LIVE_PATH_PREFIX = "/live/"
CALL_ID_PATTERN = re.compile(
    r"^(?:rtc_[A-Za-z0-9_-]+|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})$"
)

FORWARDED_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "openai-alpha",
    "originator",
    "session-id",
    "thread-id",
    "traceparent",
    "tracestate",
    "user-agent",
    "x-oai-attestation",
    "x-session-id",
}
FORWARDED_RESPONSE_HEADERS = {
    "content-type",
    "location",
    "openai-processing-ms",
    "openai-version",
    "x-request-id",
}

logger = logging.getLogger("litellm.chatgpt_live_proxy")


@dataclass(frozen=True)
class LiveProxyConfig:
    inventory_path: Path
    profile: str
    client_key: str
    call_upstream: str = DEFAULT_CALL_UPSTREAM
    ws_upstream: str = DEFAULT_WS_UPSTREAM
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LiveProxyConfig":
        inventory_path = os.getenv("CHATGPT_INVENTORY_PATH")
        client_key = os.getenv("CHATGPT_LIVE_CLIENT_KEY")
        if not inventory_path:
            raise RuntimeError("CHATGPT_INVENTORY_PATH is required")
        if not client_key:
            raise RuntimeError("CHATGPT_LIVE_CLIENT_KEY is required")
        return cls(
            inventory_path=Path(inventory_path),
            profile=os.getenv("CHATGPT_LIVE_PROFILE", DEFAULT_PROFILE),
            client_key=client_key,
            call_upstream=os.getenv(
                "CHATGPT_LIVE_CALL_UPSTREAM", DEFAULT_CALL_UPSTREAM
            ).rstrip("/"),
            ws_upstream=os.getenv(
                "CHATGPT_LIVE_WS_UPSTREAM", DEFAULT_WS_UPSTREAM
            ).rstrip("/"),
            host=os.getenv("CHATGPT_LIVE_HOST", DEFAULT_HOST),
            port=int(os.getenv("CHATGPT_LIVE_PORT", str(DEFAULT_PORT))),
            http_timeout_seconds=float(
                os.getenv(
                    "CHATGPT_LIVE_HTTP_TIMEOUT_SECONDS",
                    str(DEFAULT_HTTP_TIMEOUT_SECONDS),
                )
            ),
        )


def _configure_profiles(config: LiveProxyConfig) -> Authenticator:
    rendered = render_config(load_inventory(config.inventory_path))
    profiles = rendered["chatgpt_auth_profiles"]
    if config.profile not in profiles:
        raise RuntimeError(
            f"ChatGPT Live profile '{config.profile}' is missing or disabled"
        )
    litellm.chatgpt_auth_profiles = profiles
    return get_chatgpt_authenticator({"chatgpt_auth_profile": config.profile})


def _bearer_key(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, separator, value = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not value:
        return None
    return value


def _authorized(authorization: Optional[str], expected_key: str) -> bool:
    supplied_key = _bearer_key(authorization)
    return supplied_key is not None and hmac.compare_digest(
        supplied_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    )


def _forwarded_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in FORWARDED_REQUEST_HEADERS
    }


class LiveProxyService:
    def __init__(
        self,
        config: LiveProxyConfig,
        authenticator: Authenticator,
        http_client: httpx.AsyncClient,
        ws_connect: Callable[..., Any] = websockets.connect,
    ) -> None:
        self.config = config
        self.authenticator = authenticator
        self.http_client = http_client
        self.ws_connect = ws_connect

    async def upstream_headers(self, incoming_headers: Any) -> dict[str, str]:
        try:
            access_token = await asyncio.to_thread(
                partial(
                    self.authenticator.get_access_token,
                    allow_refresh=True,
                    allow_interactive_login=False,
                )
            )
            account_id = await asyncio.to_thread(self.authenticator.get_account_id)
        except ChatGPTAuthError:
            raise
        if not account_id:
            raise ChatGPTAuthProfileError(
                status_code=503,
                message=(
                    f"ChatGPT auth profile '{self.config.profile}' has no account ID."
                ),
            )
        headers = _forwarded_headers(incoming_headers)
        headers["Authorization"] = f"Bearer {access_token}"
        headers["ChatGPT-Account-Id"] = account_id
        return headers

    async def relay_call(self, request: Request) -> Response:
        if not _authorized(
            request.headers.get("authorization"), self.config.client_key
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            headers = await self.upstream_headers(request.headers)
        except ChatGPTAuthError as exc:
            logger.warning(
                "Live call auth failed for profile=%s: %s",
                self.config.profile,
                exc,
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "ChatGPT Live profile is unavailable"},
            )

        try:
            upstream = await self.http_client.request(
                method="POST",
                url=self.config.call_upstream,
                params=list(request.query_params.multi_items()),
                headers=headers,
                content=await request.body(),
            )
        except httpx.HTTPError as exc:
            logger.warning("Live call upstream failed: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"detail": "ChatGPT Live call upstream is unavailable"},
            )

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() in FORWARDED_RESPONSE_HEADERS
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    async def relay_websocket(self, websocket: WebSocket, call_id: str) -> None:
        if not CALL_ID_PATTERN.fullmatch(call_id):
            await websocket.close(code=1008, reason="Invalid call id")
            return
        if not _authorized(
            websocket.headers.get("authorization"), self.config.client_key
        ):
            await websocket.close(code=1008, reason="Unauthorized")
            return

        try:
            headers = await self.upstream_headers(websocket.headers)
        except ChatGPTAuthError as exc:
            logger.warning(
                "Live websocket auth failed for profile=%s: %s",
                self.config.profile,
                exc,
            )
            await websocket.close(code=1011, reason="ChatGPT profile unavailable")
            return

        requested_subprotocols = [
            value.strip()
            for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        upstream_url = f"{self.config.ws_upstream}/{call_id}"
        try:
            async with self.ws_connect(
                upstream_url,
                additional_headers=headers,
                subprotocols=requested_subprotocols or None,
                compression=None,
                max_size=None,
                user_agent_header=None,
            ) as upstream:
                selected_subprotocol = getattr(upstream, "subprotocol", None)
                await websocket.accept(subprotocol=selected_subprotocol)
                await self._relay_websocket_frames(websocket, upstream)
        except Exception as exc:
            logger.warning(
                "Live websocket upstream failed for call_id=%s: %s",
                call_id,
                exc,
            )
            try:
                await websocket.close(code=1011, reason="Live upstream unavailable")
            except RuntimeError:
                pass

    async def _relay_websocket_frames(
        self, websocket: WebSocket, upstream: Any
    ) -> None:
        async def client_to_upstream() -> None:
            while True:
                message = await websocket.receive()
                message_type = message["type"]
                if message_type == "websocket.disconnect":
                    await upstream.close(
                        code=message.get("code", 1000),
                        reason=message.get("reason", ""),
                    )
                    return
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])

        async def upstream_to_client() -> None:
            async for message in upstream:
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_bytes(message)
            await websocket.close(
                code=getattr(upstream, "close_code", None) or 1000,
                reason=getattr(upstream, "close_reason", None) or "",
            )

        tasks = {
            asyncio.create_task(client_to_upstream()),
            asyncio.create_task(upstream_to_client()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            try:
                task.result()
            except WebSocketDisconnect:
                pass


def create_app(
    config: Optional[LiveProxyConfig] = None,
    *,
    authenticator: Optional[Authenticator] = None,
    http_transport: Optional[httpx.AsyncBaseTransport] = None,
    ws_connect: Callable[..., Any] = websockets.connect,
) -> FastAPI:
    resolved_config = config or LiveProxyConfig.from_env()
    resolved_authenticator = authenticator or _configure_profiles(resolved_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(resolved_config.http_timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=http_transport,
        ) as client:
            app.state.live_proxy = LiveProxyService(
                resolved_config,
                resolved_authenticator,
                client,
                ws_connect=ws_connect,
            )
            yield

    app = FastAPI(title="ChatGPT Live Proxy", lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> Response:
        try:
            auth_data = await asyncio.to_thread(resolved_authenticator._read_auth_file)
        except ChatGPTAuthError:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready"},
            )
        if not resolved_config.client_key or not auth_data:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready"},
            )
        return JSONResponse(content={"status": "ready"})

    @app.post(CALL_PATH)
    async def call_create(request: Request) -> Response:
        return await request.app.state.live_proxy.relay_call(request)

    @app.websocket(f"{LIVE_PATH_PREFIX}{{call_id}}")
    async def live_websocket(websocket: WebSocket, call_id: str) -> None:
        await websocket.app.state.live_proxy.relay_websocket(websocket, call_id)

    return app


def main() -> None:
    config = LiveProxyConfig.from_env()
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
