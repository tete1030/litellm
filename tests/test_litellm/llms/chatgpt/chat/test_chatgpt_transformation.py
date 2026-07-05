from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


class TestChatGPTTransformation:
    @patch("litellm.llms.chatgpt.chat.transformation.get_chatgpt_authenticator")
    def test_validate_environment_uses_profile_specific_authenticator(
        self, mock_get_chatgpt_authenticator
    ):
        mock_authenticator = MagicMock()
        mock_authenticator.get_access_token.return_value = "access-123"
        mock_authenticator.get_account_id.return_value = "acct-123"
        mock_get_chatgpt_authenticator.return_value = mock_authenticator

        config = ChatGPTConfig()
        litellm_params = {
            "chatgpt_auth_profile": "account-a",
            "litellm_session_id": "session-123",
        }

        headers = config.validate_environment(
            headers={"x-test": "1"},
            model="chatgpt/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params=litellm_params,
            api_key="placeholder-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["session_id"] == "session-123"
        assert headers["x-test"] == "1"
        mock_get_chatgpt_authenticator.assert_called_with(litellm_params)

    @patch("litellm.llms.chatgpt.chat.transformation.get_chatgpt_authenticator")
    def test_validate_environment_overrides_lowercase_protected_headers(
        self, mock_get_chatgpt_authenticator
    ):
        mock_authenticator = MagicMock()
        mock_authenticator.get_access_token.return_value = "access-123"
        mock_authenticator.get_account_id.return_value = "acct-123"
        mock_get_chatgpt_authenticator.return_value = mock_authenticator

        config = ChatGPTConfig()

        headers = config.validate_environment(
            headers={
                "authorization": "Bearer wrong-token",
                "chatgpt-account-id": "wrong-acct",
                "session_id": "wrong-session",
                "originator": "custom-origin",
            },
            model="chatgpt/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={"litellm_session_id": "session-123"},
            api_key="placeholder-key",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["session_id"] == "session-123"
        assert headers["originator"] == "custom-origin"
        assert "authorization" not in headers
        assert "chatgpt-account-id" not in headers

    @patch("litellm.llms.chatgpt.chat.transformation.get_chatgpt_authenticator")
    def test_provider_info_uses_profile_specific_authenticator(
        self, mock_get_chatgpt_authenticator
    ):
        mock_authenticator = MagicMock()
        mock_authenticator.get_api_base.return_value = "https://chatgpt.example.com"
        mock_get_chatgpt_authenticator.return_value = mock_authenticator

        config = ChatGPTConfig()
        litellm_params = {"chatgpt_auth_profile": "account-a"}

        api_base, api_key, provider = config._get_openai_compatible_provider_info(
            model="chatgpt/gpt-5.4",
            api_base=None,
            api_key=None,
            custom_llm_provider="chatgpt",
            litellm_params=litellm_params,
        )

        assert api_base == "https://chatgpt.example.com"
        assert api_key == "chatgpt-oauth"
        assert provider == "chatgpt"
        mock_get_chatgpt_authenticator.assert_called_with(litellm_params)

    def test_map_openai_params_normalizes_fast_service_tier(self):
        config = ChatGPTConfig()

        optional_params = config.map_openai_params(
            non_default_params={"service_tier": "fast"},
            optional_params={},
            model="chatgpt/gpt-5.4",
            drop_params=True,
        )

        assert optional_params["service_tier"] == "priority"

    def test_transform_request_strips_fast_service_tier_for_restricted_profile(self):
        config = ChatGPTConfig()

        request = config.transform_request(
            model="chatgpt/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={"service_tier": "priority"},
            litellm_params={"chatgpt_allow_fast_mode": False},
            headers={},
        )

        assert "service_tier" not in request

    def test_transform_request_strips_fast_service_tier_for_restricted_virtual_key(
        self,
    ):
        config = ChatGPTConfig()

        request = config.transform_request(
            model="chatgpt/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={"service_tier": "priority"},
            litellm_params={
                "chatgpt_allow_fast_mode": True,
                "metadata": {
                    "user_api_key_auth_metadata": {
                        "chatgpt_virtual_key_allow_fast_mode": False
                    }
                },
            },
            headers={},
        )

        assert "service_tier" not in request
