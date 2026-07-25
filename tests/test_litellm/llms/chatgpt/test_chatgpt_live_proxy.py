import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from litellm.llms.chatgpt.live_proxy import (
    CALL_PATH,
    LiveProxyConfig,
    create_app,
)


class FakeUpstreamWebSocket:
    def __init__(self) -> None:
        self.subprotocol = None
        self.close_code = 1000
        self.close_reason = ""
        self.sent: list[Any] = []
        self._messages: asyncio.Queue[Any] = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._messages.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def send(self, message: Any) -> None:
        self.sent.append(message)
        await self._messages.put(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.close_reason = reason
        await self._messages.put(None)


class FakeWebSocketConnection:
    def __init__(self, upstream: FakeUpstreamWebSocket) -> None:
        self.upstream = upstream

    async def __aenter__(self) -> FakeUpstreamWebSocket:
        return self.upstream

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.upstream.close()


class FakeWebSocketConnector:
    def __init__(self) -> None:
        self.upstream = FakeUpstreamWebSocket()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeWebSocketConnection:
        self.calls.append((url, kwargs))
        return FakeWebSocketConnection(self.upstream)


@pytest.fixture
def config(tmp_path: Path) -> LiveProxyConfig:
    return LiveProxyConfig(
        inventory_path=tmp_path / "inventory.yaml",
        profile="my",
        client_key="sk-test-live",
        call_upstream="https://chatgpt.test/backend-api/codex/realtime/calls",
        ws_upstream="wss://api.openai.test/v1/live",
    )


@pytest.fixture
def authenticator() -> MagicMock:
    auth = MagicMock()
    auth.get_access_token.return_value = "oauth-access-token"
    auth.get_account_id.return_value = "acct-test"
    auth._read_auth_file.return_value = {
        "access_token": "oauth-access-token",
        "account_id": "acct-test",
    }
    return auth


def test_health_and_readiness(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    app = create_app(config, authenticator=authenticator)

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}

    authenticator.get_access_token.assert_not_called()


def test_call_create_replaces_auth_and_preserves_response(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    captured: dict[str, Any] = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            201,
            content=b"v=0\r\ns=answer\r\n",
            headers={
                "Content-Type": "application/sdp",
                "Location": "https://api.openai.com/v1/live/rtc_test",
                "X-Request-Id": "req-test",
                "Set-Cookie": "must-not-pass=true",
            },
        )

    app = create_app(
        config,
        authenticator=authenticator,
        http_transport=httpx.MockTransport(upstream),
    )

    with TestClient(app) as client:
        response = client.post(
            f"{CALL_PATH}?intent=quicksilver&architecture=avas",
            headers={
                "Authorization": "Bearer sk-test-live",
                "Content-Type": "application/json",
                "x-oai-attestation": "attestation-test",
                "x-session-id": "session-test",
            },
            content=json.dumps({"sdp": "offer", "session": {"type": "live"}}),
        )

    request = captured["request"]
    assert response.status_code == 201
    assert response.content == b"v=0\r\ns=answer\r\n"
    assert response.headers["location"].endswith("/rtc_test")
    assert response.headers["x-request-id"] == "req-test"
    assert "set-cookie" not in response.headers
    assert request.headers["authorization"] == "Bearer oauth-access-token"
    assert request.headers["chatgpt-account-id"] == "acct-test"
    assert request.headers["x-oai-attestation"] == "attestation-test"
    assert request.headers["x-session-id"] == "session-test"
    assert request.url.params["intent"] == "quicksilver"
    assert request.url.params["architecture"] == "avas"
    authenticator.get_access_token.assert_called_once_with(
        allow_refresh=True,
        allow_interactive_login=False,
    )


def test_call_create_rejects_invalid_client_key(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    app = create_app(config, authenticator=authenticator)

    with TestClient(app) as client:
        response = client.post(
            CALL_PATH,
            headers={"Authorization": "Bearer wrong-key"},
            content=b"{}",
        )

    assert response.status_code == 401
    authenticator.get_access_token.assert_not_called()


def test_call_create_rejects_profile_without_account_id(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    authenticator.get_account_id.return_value = None
    app = create_app(config, authenticator=authenticator)

    with TestClient(app) as client:
        response = client.post(
            CALL_PATH,
            headers={"Authorization": "Bearer sk-test-live"},
            content=b"{}",
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "ChatGPT Live profile is unavailable"}


def test_websocket_relays_text_and_binary_and_replaces_auth(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    connector = FakeWebSocketConnector()
    app = create_app(
        config,
        authenticator=authenticator,
        ws_connect=connector,
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/live/rtc_test",
            headers={
                "Authorization": "Bearer sk-test-live",
                "x-oai-attestation": "attestation-test",
                "x-session-id": "session-test",
            },
        ) as websocket:
            websocket.send_text("hello")
            assert websocket.receive_text() == "hello"
            websocket.send_bytes(b"\x00\x01\x02")
            assert websocket.receive_bytes() == b"\x00\x01\x02"

    assert connector.calls
    url, kwargs = connector.calls[0]
    assert url == "wss://api.openai.test/v1/live/rtc_test"
    assert kwargs["additional_headers"]["Authorization"] == (
        "Bearer oauth-access-token"
    )
    assert kwargs["additional_headers"]["ChatGPT-Account-Id"] == "acct-test"
    assert kwargs["additional_headers"]["x-oai-attestation"] == "attestation-test"
    assert kwargs["user_agent_header"] is None
    assert connector.upstream.sent == ["hello", b"\x00\x01\x02"]


def test_websocket_rejects_invalid_call_id(
    config: LiveProxyConfig, authenticator: MagicMock
) -> None:
    connector = FakeWebSocketConnector()
    app = create_app(
        config,
        authenticator=authenticator,
        ws_connect=connector,
    )

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/live/invalid",
                headers={"Authorization": "Bearer sk-test-live"},
            ):
                pass

    assert connector.calls == []
    authenticator.get_access_token.assert_not_called()
