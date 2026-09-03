from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.proxy.middleware import traffic_metrics
from litellm.types.router import GenericLiteLLMParams


def _mock_responses_config():
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example/responses"
    config.transform_responses_api_request.return_value = {"model": "gpt-5"}
    config.transform_response_api_response.return_value = "ok"
    return config


def _mock_logging_obj():
    logging_obj = Mock()
    logging_obj.model_call_details = {"litellm_params": {}}
    return logging_obj


@pytest.mark.parametrize("stream", [False, True])
def test_response_api_handler_passes_logging_object_to_http_handler(stream):
    handler = BaseLLMHTTPHandler()
    client = Mock(spec=HTTPHandler)
    client.post.return_value = httpx.Response(200, content=b"")
    logging_obj = _mock_logging_obj()

    handler.response_api_handler(
        model="gpt-5",
        input=[{"role": "user", "content": "hello"}],
        responses_api_provider_config=_mock_responses_config(),
        response_api_optional_request_params={"stream": stream},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["logging_obj"] is logging_obj


@pytest.mark.asyncio
async def test_async_response_api_handler_records_chatgpt_provider_traffic(
    monkeypatch,
):
    prometheus_logger = Mock()
    monkeypatch.setattr(
        traffic_metrics, "_get_prometheus_logger", lambda: prometheus_logger
    )
    monkeypatch.setattr(
        traffic_metrics, "_schedule_daily_traffic_aggregate", lambda **kwargs: None
    )

    async def upstream(request):
        assert request.content == b'{"model":"gpt-5"}'
        return httpx.Response(200, json={"id": "response-id"})

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    logging_obj = _mock_logging_obj()
    logging_obj.model_call_details = {
        "custom_llm_provider": "chatgpt",
        "model": "gpt-5",
        "litellm_params": {
            "metadata": {
                "user_api_key_hash": "key-hash",
                "user_api_key_alias": "smoke",
                "user_api_key_request_route": "/v1/responses",
                "model_group": "gpt-5",
            }
        },
    }

    try:
        result = await BaseLLMHTTPHandler().async_response_api_handler(
            model="gpt-5",
            input=[{"role": "user", "content": "hello"}],
            responses_api_provider_config=_mock_responses_config(),
            response_api_optional_request_params={"stream": False},
            custom_llm_provider="chatgpt",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=logging_obj,
            client=client,
        )
    finally:
        await client.close()

    assert result == "ok"
    prometheus_logger.observe_provider_body_bytes.assert_called_once()
    event = prometheus_logger.observe_provider_body_bytes.call_args.kwargs
    assert event["api_provider"] == "chatgpt"
    assert event["request_body_bytes"] == len(b'{"model":"gpt-5"}')
    assert event["response_body_bytes"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_response_api_handler_passes_logging_object_to_http_handler(
    stream,
):
    handler = BaseLLMHTTPHandler()
    client = AsyncMock(spec=AsyncHTTPHandler)
    client.post.return_value = httpx.Response(200, content=b"")
    logging_obj = _mock_logging_obj()

    await handler.async_response_api_handler(
        model="gpt-5",
        input=[{"role": "user", "content": "hello"}],
        responses_api_provider_config=_mock_responses_config(),
        response_api_optional_request_params={"stream": stream},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["logging_obj"] is logging_obj
