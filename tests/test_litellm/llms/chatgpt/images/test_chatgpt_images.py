import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.images.main import aimage_edit, image_edit, image_generation
from litellm.llms.chatgpt.images.auth import (
    ChatGPTImageAuth,
    resolve_chatgpt_image_auth,
)
from litellm.llms.chatgpt.images.transformation import ChatGPTImageEditConfig
from litellm.types.router import GenericLiteLLMParams


class _FakeAuthenticator:
    def get_access_token(self) -> str:
        return "real-oauth-token"

    def get_account_id(self) -> str:
        return "account-123"

    def get_api_base(self) -> str:
        return "https://chatgpt.com/backend-api/codex/"


def _resolved_auth() -> ChatGPTImageAuth:
    return ChatGPTImageAuth(
        api_base="https://chatgpt.com/backend-api/codex",
        api_key="real-oauth-token",
        headers={
            "Authorization": "Bearer real-oauth-token",
            "ChatGPT-Account-Id": "account-123",
            "session_id": "session-123",
            "content-type": "application/json",
            "accept": "application/json",
            "originator": "codex_cli_rs",
            "user-agent": "codex_cli_rs/0.144.1",
        },
    )


@patch(
    "litellm.llms.chatgpt.images.auth.get_chatgpt_authenticator",
    return_value=_FakeAuthenticator(),
)
def test_resolve_chatgpt_image_auth_uses_profile_and_protects_headers(
    mock_get_authenticator: MagicMock,
) -> None:
    params = {
        "chatgpt_auth_profile": "account-a",
        "litellm_session_id": "session-123",
        "metadata": {"user_agent": "codex_cli_rs/0.144.1"},
    }
    result = resolve_chatgpt_image_auth(
        litellm_params=params,
        headers={
            "authorization": "Bearer caller-token",
            "Accept": "text/event-stream",
            "Originator": "caller",
            "X-Trace": "trace-123",
        },
    )

    mock_get_authenticator.assert_called_once_with(params)
    assert result.api_key == "real-oauth-token"
    assert result.api_base == "https://chatgpt.com/backend-api/codex"
    assert result.headers["Authorization"] == "Bearer real-oauth-token"
    assert result.headers["ChatGPT-Account-Id"] == "account-123"
    assert result.headers["session_id"] == "session-123"
    assert result.headers["accept"] == "application/json"
    assert result.headers["originator"] == "codex_cli_rs"
    assert result.headers["user-agent"].startswith("codex_cli_rs/")
    assert result.headers["X-Trace"] == "trace-123"
    assert "authorization" not in result.headers
    assert "Accept" not in result.headers
    assert "Originator" not in result.headers


@patch("litellm.images.main.openai_chat_completions")
@patch(
    "litellm.llms.chatgpt.images.auth.resolve_chatgpt_image_auth",
    return_value=_resolved_auth(),
)
def test_chatgpt_image_generation_uses_native_oauth(
    mock_resolve_auth: MagicMock,
    mock_openai_chat_completions: MagicMock,
) -> None:
    expected_response = litellm.utils.ImageResponse(
        created=123,
        data=[{"b64_json": "image-data"}],
    )
    mock_openai_chat_completions.image_generation.return_value = expected_response

    response = image_generation(
        model="chatgpt/gpt-image-2",
        prompt="Draw a red circle",
        chatgpt_auth_profile="account-a",
        client=MagicMock(),
        extra_headers={"X-Trace": "trace-123"},
    )

    assert response == expected_response
    auth_params = mock_resolve_auth.call_args.kwargs["litellm_params"]
    assert auth_params["chatgpt_auth_profile"] == "account-a"

    call_kwargs = mock_openai_chat_completions.image_generation.call_args.kwargs
    assert call_kwargs["api_key"] == "real-oauth-token"
    assert call_kwargs["api_base"] == "https://chatgpt.com/backend-api/codex"
    assert call_kwargs["client"] is None
    assert call_kwargs["headers"] == _resolved_auth().headers
    assert call_kwargs["optional_params"]["extra_headers"] == _resolved_auth().headers
    assert "chatgpt_auth_profile" not in call_kwargs["optional_params"]


def test_chatgpt_image_edit_transforms_data_urls_and_bytes() -> None:
    config = ChatGPTImageEditConfig()
    original_data_url = "data:image/png;base64,AAAA"
    raw_image = b"\x89PNG\r\n\x1a\nraw-image"

    data, files = config.transform_image_edit_request(
        model="gpt-image-2",
        prompt="Make the circle blue",
        image=[
            {"image_url": original_data_url},
            BytesIO(raw_image),
            ("photo.png", raw_image, "image/png"),
        ],
        image_edit_optional_request_params={
            "background": "auto",
            "quality": "low",
            "size": "auto",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert config.use_multipart_form_data() is False
    assert list(files) == []
    assert data["model"] == "gpt-image-2"
    assert data["prompt"] == "Make the circle blue"
    assert data["background"] == "auto"
    assert data["quality"] == "low"
    assert data["size"] == "auto"
    assert data["images"][0] == {"image_url": original_data_url}
    expected_payload = base64.b64encode(raw_image).decode("ascii")
    assert data["images"][1] == {"image_url": f"data:image/png;base64,{expected_payload}"}
    assert data["images"][2] == {"image_url": f"data:image/png;base64,{expected_payload}"}


@patch("litellm.images.main.base_llm_http_handler")
@patch(
    "litellm.llms.chatgpt.images.auth.resolve_chatgpt_image_auth",
    return_value=_resolved_auth(),
)
def test_chatgpt_image_edit_routes_json_images_alias(
    mock_resolve_auth: MagicMock,
    mock_http_handler: MagicMock,
) -> None:
    expected_response = litellm.utils.ImageResponse(
        created=123,
        data=[{"b64_json": "edited-image"}],
    )
    mock_http_handler.image_edit_handler.return_value = expected_response
    image_url = "data:image/png;base64,AAAA"

    response = image_edit(
        model="chatgpt/gpt-image-2",
        prompt="Make the circle blue",
        images=[{"image_url": image_url}],
        chatgpt_auth_profile="account-a",
        extra_headers={"Authorization": "Bearer caller-token"},
    )

    assert response == expected_response
    auth_params = mock_resolve_auth.call_args.kwargs["litellm_params"]
    assert auth_params.chatgpt_auth_profile == "account-a"

    call_kwargs = mock_http_handler.image_edit_handler.call_args.kwargs
    assert isinstance(call_kwargs["image_edit_provider_config"], ChatGPTImageEditConfig)
    assert call_kwargs["image"] == [{"image_url": image_url}]
    assert call_kwargs["litellm_params"].api_key == "real-oauth-token"
    assert call_kwargs["litellm_params"].api_base == "https://chatgpt.com/backend-api/codex"
    assert call_kwargs["extra_headers"] == _resolved_auth().headers


@pytest.mark.asyncio
@patch("litellm.images.main.base_llm_http_handler")
@patch(
    "litellm.llms.chatgpt.images.auth.resolve_chatgpt_image_auth",
    return_value=_resolved_auth(),
)
async def test_chatgpt_async_image_edit_accepts_codex_images_alias(
    mock_resolve_auth: MagicMock,
    mock_http_handler: MagicMock,
) -> None:
    expected_response = litellm.utils.ImageResponse(
        created=123,
        data=[{"b64_json": "edited-image"}],
    )
    mock_http_handler.image_edit_handler.return_value = expected_response
    image_url = "data:image/png;base64,AAAA"

    response = await aimage_edit(
        model="chatgpt/gpt-image-2",
        prompt="Make the circle blue",
        images=[{"image_url": image_url}],
        chatgpt_auth_profile="account-a",
    )

    assert response == expected_response
    call_kwargs = mock_http_handler.image_edit_handler.call_args.kwargs
    assert call_kwargs["image"] == [{"image_url": image_url}]
    assert call_kwargs["_is_async"] is True
