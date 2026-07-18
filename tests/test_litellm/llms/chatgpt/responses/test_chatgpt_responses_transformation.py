"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""
import json
import os
import sys
from typing import cast
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_get_chatgpt_authenticator):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"
        mock_get_chatgpt_authenticator.assert_called_with({})

    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_validate_environment_headers(self, mock_get_chatgpt_authenticator):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = cast(
            GenericLiteLLMParams, {"litellm_session_id": "session-123"}
        )
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] == "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"
        mock_get_chatgpt_authenticator.assert_called_with(litellm_params)

    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_validate_environment_forwards_codex_user_agent(
        self, mock_get_chatgpt_authenticator
    ):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="gpt-5.6-sol",
            litellm_params=cast(
                GenericLiteLLMParams,
                {
                    "metadata": {
                        "user_agent": "codex_cli_rs/0.144.1 (Linux 6; x86_64) unknown"
                    }
                },
            ),
        )

        assert headers["user-agent"] == "codex_cli_rs/0.144.1 (Linux 6; x86_64) unknown"

    @patch.dict(
        os.environ,
        {"CHATGPT_CODEX_CLIENT_VERSION": "0.144.1"},
        clear=False,
    )
    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_validate_environment_replaces_non_codex_user_agent(
        self, mock_get_chatgpt_authenticator
    ):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        headers = config.validate_environment(
            headers={"User-Agent": "opencode/1.2.3"},
            model="gpt-5.6-sol",
            litellm_params=cast(
                GenericLiteLLMParams,
                {"metadata": {"user_agent": "opencode/1.2.3"}},
            ),
        )

        assert headers["user-agent"].startswith("codex_cli_rs/0.144.1 ")
        assert "User-Agent" not in headers

    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_validate_environment_uses_profile_specific_authenticator(
        self, mock_get_chatgpt_authenticator
    ):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "profile-token"
        mock_auth_instance.get_account_id.return_value = "acct-profile"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = cast(
            GenericLiteLLMParams,
            {
                "chatgpt_auth_profile": "account-a",
                "litellm_session_id": "session-a",
            },
        )

        headers = config.validate_environment(
            headers={},
            model="chatgpt/gpt-5.3-codex",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer profile-token"
        assert headers["ChatGPT-Account-Id"] == "acct-profile"
        mock_get_chatgpt_authenticator.assert_called_with(litellm_params)

    @patch("litellm.llms.chatgpt.responses.transformation.get_chatgpt_authenticator")
    def test_validate_environment_overrides_lowercase_protected_headers(
        self, mock_get_chatgpt_authenticator
    ):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "profile-token"
        mock_auth_instance.get_account_id.return_value = "acct-profile"
        mock_get_chatgpt_authenticator.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        headers = config.validate_environment(
            headers={
                "authorization": "Bearer wrong-token",
                "chatgpt-account-id": "wrong-acct",
                "accept": "application/json",
                "originator": "custom-origin",
            },
            model="chatgpt/gpt-5.3-codex",
            litellm_params=cast(
                GenericLiteLLMParams, {"litellm_session_id": "session-a"}
            ),
        )

        assert headers["Authorization"] == "Bearer profile-token"
        assert headers["ChatGPT-Account-Id"] == "acct-profile"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-a"
        assert headers["originator"] == "custom-origin"
        assert "authorization" not in headers
        assert "chatgpt-account-id" not in headers

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 200000}
                ],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }

    def test_chatgpt_normalizes_fast_service_tier_to_priority(self):
        config = ChatGPTResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.3-codex",
            input="hi",
            response_api_optional_request_params={"service_tier": "fast"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["service_tier"] == "priority"

    def test_chatgpt_strips_fast_service_tier_for_restricted_profile(self):
        config = ChatGPTResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.3-codex",
            input="hi",
            response_api_optional_request_params={"service_tier": "priority"},
            litellm_params=GenericLiteLLMParams(chatgpt_allow_fast_mode=False),
            headers={},
        )

        assert "service_tier" not in request

    def test_chatgpt_strips_fast_service_tier_for_restricted_virtual_key(self):
        config = ChatGPTResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.3-codex",
            input="hi",
            response_api_optional_request_params={"service_tier": "priority"},
            litellm_params=GenericLiteLLMParams(
                chatgpt_allow_fast_mode=True,
                metadata={
                    "user_api_key_auth_metadata": {
                        "chatgpt_virtual_key_allow_fast_mode": False
                    }
                },
            ),
            headers={},
        )

        assert "service_tier" not in request

    def test_chatgpt_replaces_reasoning_effort_for_virtual_key(self):
        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(
            metadata={
                "user_api_key_alias": "vk-reasoning",
                "user_api_key_auth_metadata": {
                    "chatgpt_reasoning_effort_policy": {
                        "models": {
                            "gpt-5.6-sol": {
                                "levels": {
                                    "max": {
                                        "action": "replace",
                                        "target": "xhigh",
                                    }
                                }
                            }
                        }
                    }
                },
            }
        )

        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.6-sol",
            input="hi",
            response_api_optional_request_params={
                "reasoning": {"effort": "max", "summary": "detailed"}
            },
            litellm_params=litellm_params,
            headers={},
        )

        assert request["reasoning"] == {
            "effort": "xhigh",
            "summary": "detailed",
        }
        assert litellm_params.metadata["chatgpt_requested_reasoning_effort"] == "max"
        assert (
            litellm_params.metadata["chatgpt_effective_reasoning_effort"]
            == "xhigh"
        )

    def test_chatgpt_rejects_reasoning_effort_for_virtual_key(self):
        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(
            metadata={
                "user_api_key_auth_metadata": {
                    "chatgpt_reasoning_effort_policy": {
                        "models": {
                            "gpt-5.6-sol": {
                                "levels": {"ultra": {"action": "reject"}}
                            }
                        }
                    }
                }
            }
        )

        with pytest.raises(UnsupportedParamsError, match="is not allowed"):
            config.transform_responses_api_request(
                model="chatgpt/gpt-5.6-sol",
                input="hi",
                response_api_optional_request_params={
                    "reasoning": {"effort": "ultra"}
                },
                litellm_params=litellm_params,
                headers={},
            )

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    def test_chatgpt_non_stream_sse_uses_output_item_done_when_completed_output_empty(
        self,
    ):
        config = ChatGPTResponsesAPIConfig()
        done_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Recovered text"}],
        }
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': done_item})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.3-codex",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert parsed.output_text == "Recovered text"
        assert parsed.output == [done_item]

    def test_chatgpt_non_stream_sse_prefers_completed_output_over_done_items(self):
        config = ChatGPTResponsesAPIConfig()
        done_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Stale text"}],
        }
        completed_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Authoritative text"}],
        }
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [completed_item],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': done_item})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.3-codex",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert parsed.output_text == "Authoritative text"
        assert parsed.output == [completed_item]
