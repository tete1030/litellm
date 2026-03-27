import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm

from litellm.proxy.proxy_server import ProxyConfig


def test_proxy_validation_accepts_implicit_default_profile():
    proxy_config = ProxyConfig()
    litellm.chatgpt_auth_profiles = {}

    proxy_config.validate_chatgpt_auth_profile_references(
        model_list=[
            {
                "model_name": "codex",
                "litellm_params": {
                    "model": "chatgpt/gpt-5.3-codex",
                    "chatgpt_auth_profile": "default",
                },
            }
        ],
        chatgpt_auth_profiles={},
    )


def test_proxy_validation_accepts_env_defined_profile(monkeypatch):
    proxy_config = ProxyConfig()
    litellm.chatgpt_auth_profiles = {}
    monkeypatch.setenv(
        "CHATGPT_AUTH_PROFILES_JSON",
        '{"account-b": {"token_dir": "/tmp/chatgpt-account-b"}}',
    )

    proxy_config.validate_chatgpt_auth_profile_references(
        model_list=[
            {
                "model_name": "codex",
                "litellm_params": {
                    "model": "chatgpt/gpt-5.3-codex",
                    "chatgpt_auth_profile": "account-b",
                },
            }
        ],
        chatgpt_auth_profiles={},
    )


def test_proxy_validation_rejects_named_profile_colliding_with_implicit_default(
    monkeypatch,
):
    proxy_config = ProxyConfig()
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", "/tmp/chatgpt-default")
    litellm.chatgpt_auth_profiles = {
        "account-a": {"token_dir": "/tmp/chatgpt-default"}
    }

    with pytest.raises(ValueError, match="implicit default profile"):
        proxy_config.validate_chatgpt_auth_profile_references(
            model_list=[
                {
                    "model_name": "codex",
                    "litellm_params": {
                        "model": "chatgpt/gpt-5.3-codex",
                        "chatgpt_auth_profile": "account-a",
                    },
                }
            ],
            chatgpt_auth_profiles=litellm.chatgpt_auth_profiles,
        )


@patch("litellm.proxy.proxy_server.ProxyConfig._init_non_llm_configs", new_callable=AsyncMock)
@patch("litellm.proxy.proxy_server.ProxyConfig._init_policy_engine", new_callable=AsyncMock)
@patch("litellm.proxy.proxy_server.ProxyConfig.get_config", new_callable=AsyncMock)
@patch("litellm.proxy.proxy_server.litellm.Router")
def test_proxy_load_config_sets_registry_before_validation(
    mock_router_class,
    mock_get_config,
    mock_init_policy_engine,
    mock_init_non_llm_configs,
):
    config = {
        "model_list": [],
    }
    expected_profiles = {
        "account-a": {
            "token_dir": "/tmp/chatgpt-account-a",
            "auth_file": "/tmp/chatgpt-account-a/auth.json",
        }
    }

    class TrackingProxyConfig(ProxyConfig):
        def __init__(self):
            super().__init__()
            self.validate_called = False
            self.load_profiles_called = False

        def load_chatgpt_auth_profiles(self, config):
            self.load_profiles_called = True
            assert config["model_list"] == []
            return expected_profiles

        def validate_chatgpt_auth_profile_references(
            self, model_list, chatgpt_auth_profiles
        ):
            self.validate_called = True
            assert self.load_profiles_called is True
            assert litellm.chatgpt_auth_profiles == expected_profiles
            assert chatgpt_auth_profiles == expected_profiles

    proxy_config = TrackingProxyConfig()

    mock_get_config.return_value = config
    mock_router = MagicMock()
    mock_router.get_model_list.return_value = []
    mock_router.cache.redis_cache = None
    mock_router_class.return_value = mock_router

    asyncio.run(
        proxy_config.load_config(
            router=None,
            config_file_path="",
        )
    )

    assert proxy_config.validate_called is True
    assert proxy_config.load_profiles_called is True
