import httpx
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.middleware import traffic_metrics
from litellm.proxy.middleware.traffic_metrics import (
    ProviderTrafficObserver,
    TrafficMetricsMiddleware,
)


class _RecordingPrometheusLogger:
    def __init__(self):
        self.client_events = []
        self.provider_events = []

    def observe_client_body_bytes(self, **kwargs):
        self.client_events.append(kwargs)

    def observe_provider_body_bytes(self, **kwargs):
        self.provider_events.append(kwargs)


class _AsyncStreamingBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"data: first\n\n"
        yield b"data: second\n\n"

    async def aclose(self):
        return None


def test_daily_aggregate_date_uses_configured_timezone(monkeypatch):
    monkeypatch.setenv("LITELLM_TRAFFIC_METRICS_DAILY_TIMEZONE", "Asia/Shanghai")

    assert (
        traffic_metrics._daily_aggregate_date()
        == datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    )


@pytest.mark.asyncio
async def test_middleware_counts_request_and_response_bodies(monkeypatch):
    logger = _RecordingPrometheusLogger()
    monkeypatch.setattr(traffic_metrics, "_get_prometheus_logger", lambda: logger)

    async def app(scope, receive, send):
        received = await receive()
        assert received["body"] == b'{"model":"test"}'
        scope["state"][traffic_metrics.TRAFFIC_IDENTITY_SCOPE_KEY] = {
            "hashed_api_key": "key-hash",
            "api_key_alias": "team-a",
            "route": "/v1/responses",
            "requested_model": "test",
            "stream": True,
        }
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first"})
        await send({"type": "http.response.body", "body": b"second"})

    messages = [
        {"type": "http.request", "body": b'{"model":"test"}', "more_body": False}
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        return None

    middleware = TrafficMetricsMiddleware(app)
    await middleware(
        {"type": "http", "path": "/v1/responses", "state": {}}, receive, send
    )

    assert logger.client_events == [
        {
            "hashed_api_key": "key-hash",
            "api_key_alias": "team-a",
            "requested_model": "test",
            "route": "/v1/responses",
            "status_code": 200,
            "stream": True,
            "request_body_bytes": len(b'{"model":"test"}'),
            "response_body_bytes": len(b"firstsecond"),
        }
    ]


@pytest.mark.asyncio
async def test_chatgpt_handler_counts_provider_stream_after_consumption(monkeypatch):
    logger = _RecordingPrometheusLogger()
    monkeypatch.setattr(traffic_metrics, "_get_prometheus_logger", lambda: logger)

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.content == b'{"model":"gpt-5"}'
        return httpx.Response(200, stream=_AsyncStreamingBody())

    logging_obj = type(
        "LoggingObject",
        (),
        {
            "model_call_details": {
                "custom_llm_provider": "chatgpt",
                "model": "gpt-5",
                "traffic_provider_attempt": 2,
                "litellm_params": {
                    "metadata": {
                        "user_api_key_hash": "key-hash",
                        "user_api_key_alias": "team-a",
                        "user_api_key_request_route": "/v1/responses",
                        "model_group": "gpt-5",
                        "model_info": {"id": "chatgpt-primary"},
                        "chatgpt_auth_profile": "primary",
                    }
                },
            }
        },
    )()
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await handler.post(
            "https://chatgpt.example/responses",
            data='{"model":"gpt-5"}',
            stream=True,
            logging_obj=logging_obj,
        )
        assert logger.provider_events == []
        assert await response.aread() == b"data: first\n\ndata: second\n\n"
    finally:
        await handler.client.aclose()

    assert logger.provider_events == [
        {
            "hashed_api_key": "key-hash",
            "api_key_alias": "team-a",
            "requested_model": "gpt-5",
            "route": "/v1/responses",
            "model": "gpt-5",
            "model_id": "chatgpt-primary",
            "api_provider": "chatgpt",
            "chatgpt_auth_profile": "primary",
            "status_code": 200,
            "attempt": 2,
            "stream": True,
            "request_body_bytes": len(b'{"model":"gpt-5"}'),
            "response_body_bytes": len(b"data: first\n\ndata: second\n\n"),
        }
    ]


def test_provider_observer_passes_only_the_key_hash_to_daily_aggregation(monkeypatch):
    scheduled_events = []
    monkeypatch.setattr(
        traffic_metrics,
        "_schedule_daily_traffic_aggregate",
        lambda **kwargs: scheduled_events.append(kwargs),
    )
    monkeypatch.setattr(traffic_metrics, "_get_prometheus_logger", lambda: None)

    observer = ProviderTrafficObserver(
        hashed_api_key=None,
        api_key_alias=None,
        requested_model="gpt-5",
        route="/v1/responses",
        model="gpt-5",
        model_id="deployment",
        api_provider="chatgpt",
        chatgpt_auth_profile="primary",
        attempt=1,
        stream=False,
    )
    observer.record_request_body(b"request")
    observer.finish(200)

    assert scheduled_events == [
        {
            "hashed_api_key": None,
            "api_key_alias": None,
            "route": "/v1/responses",
            "requested_model": "gpt-5",
            "provider_request_body_bytes": len(b"request"),
            "provider_response_body_bytes": 0,
            "provider_attempts": 1,
        }
    ]


@pytest.mark.asyncio
async def test_daily_aggregate_upsert_contains_only_aggregate_fields(monkeypatch):
    calls = []

    class _Table:
        async def upsert(self, **kwargs):
            calls.append(kwargs)

    table = _Table()
    proxy_server = type(
        "ProxyServer",
        (),
        {
            "prisma_client": type(
                "PrismaClient",
                (),
                {"db": type("DB", (), {"litellm_dailytraffic": table})()},
            )()
        },
    )()
    import litellm.proxy as proxy_package

    monkeypatch.setattr(proxy_package, "proxy_server", proxy_server, raising=False)
    await traffic_metrics._upsert_daily_traffic_aggregate(
        hashed_api_key="key-hash",
        api_key_alias="team-a",
        route="/v1/responses",
        requested_model="gpt-5",
        client_request_body_bytes=12,
        client_response_body_bytes=34,
        provider_request_body_bytes=56,
        provider_response_body_bytes=78,
        client_requests=1,
        provider_attempts=2,
    )

    assert len(calls) == 1
    payload = calls[0]
    assert payload["where"]["date_api_key_route_requested_model"] == {
        "date": payload["data"]["create"]["date"],
        "api_key": "key-hash",
        "route": "/v1/responses",
        "requested_model": "gpt-5",
    }
    assert payload["data"]["create"] == {
        "date": payload["data"]["create"]["date"],
        "api_key": "key-hash",
        "api_key_alias": "team-a",
        "route": "/v1/responses",
        "requested_model": "gpt-5",
        "client_request_body_bytes": 12,
        "client_response_body_bytes": 34,
        "provider_request_body_bytes": 56,
        "provider_response_body_bytes": 78,
        "client_requests": 1,
        "provider_attempts": 2,
    }
