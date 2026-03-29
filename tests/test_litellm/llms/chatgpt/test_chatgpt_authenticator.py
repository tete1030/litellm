import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.chatgpt.authenticator import (
    Authenticator,
    get_chatgpt_authenticator,
    reset_chatgpt_authenticator_cache,
    resolve_chatgpt_auth_profile,
)
from litellm.llms.chatgpt.common_utils import (
    ChatGPTAuthProfileError,
    GetAccessTokenError,
)
from litellm.types.router import GenericLiteLLMParams


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


def _write_auth_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture(autouse=True)
def _reset_chatgpt_auth_state(monkeypatch):
    litellm.chatgpt_auth_profiles = {}
    reset_chatgpt_authenticator_cache()
    for env_var in (
        "CHATGPT_TOKEN_DIR",
        "CHATGPT_AUTH_FILE",
        "CHATGPT_AUTH_PROFILES",
        "CHATGPT_AUTH_PROFILES_JSON",
    ):
        monkeypatch.delenv(env_var, raising=False)
    yield
    litellm.chatgpt_auth_profiles = {}
    reset_chatgpt_authenticator_cache()


class TestChatGPTAuthenticator:
    def test_get_access_token_from_default_profile_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "chatgpt-auth.json")
        future_time = time.time() + 3600
        auth_path = tmp_path / "chatgpt-auth.json"
        _write_auth_file(
            auth_path,
            {"access_token": "token-123", "expires_at": future_time},
        )

        authenticator = Authenticator()

        assert authenticator.auth_file == str(auth_path)
        assert authenticator.get_access_token() == "token-123"

    def test_named_profile_resolution_and_authenticator_cache(self, tmp_path):
        litellm.chatgpt_auth_profiles = {
            "account-a": {"token_dir": str(tmp_path / "account-a")},
            "account-b": {"auth_file": str(tmp_path / "account-b" / "custom.json")},
        }

        auth_a = get_chatgpt_authenticator(
            {"chatgpt_auth_profile": "account-a"}
        )
        auth_a_again = get_chatgpt_authenticator(
            GenericLiteLLMParams(chatgpt_auth_profile="account-a")
        )
        auth_b = get_chatgpt_authenticator(
            {"chatgpt_auth_profile": "account-b"}
        )

        assert auth_a is auth_a_again
        assert auth_a is not auth_b
        assert auth_a.profile_name == "account-a"
        assert auth_b.profile_name == "account-b"
        assert auth_b.auth_file == str(tmp_path / "account-b" / "custom.json")

    def test_legacy_chatgpt_auth_file_does_not_leak_to_named_profiles(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "/tmp/legacy-auth.json")
        litellm.chatgpt_auth_profiles = {
            "account-a": {"token_dir": str(tmp_path / "account-a")},
            "account-b": {"token_dir": str(tmp_path / "account-b")},
        }

        auth_a = resolve_chatgpt_auth_profile(profile_name="account-a")
        auth_b = resolve_chatgpt_auth_profile(profile_name="account-b")

        assert auth_a.auth_file == str(tmp_path / "account-a" / "auth.json")
        assert auth_b.auth_file == str(tmp_path / "account-b" / "auth.json")

    def test_duplicate_profile_auth_files_raise_error(self, tmp_path):
        with pytest.raises(ChatGPTAuthProfileError, match="same auth_file"):
            litellm.chatgpt_auth_profiles = {
                "account-a": {"auth_file": str(tmp_path / "shared.json")},
                "account-b": {"auth_file": str(tmp_path / "shared.json")},
            }
            resolve_chatgpt_auth_profile(profile_name="account-a")

    def test_named_profile_cannot_share_auth_file_with_implicit_default(
        self, monkeypatch, tmp_path
    ):
        shared_dir = tmp_path / "shared"
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(shared_dir))
        litellm.chatgpt_auth_profiles = {
            "account-a": {"token_dir": str(shared_dir)}
        }

        with pytest.raises(
            ChatGPTAuthProfileError, match="implicit default profile"
        ):
            resolve_chatgpt_auth_profile(profile_name="account-a")

    def test_unknown_profile_raises_actionable_error(self):
        with pytest.raises(ChatGPTAuthProfileError, match="Unknown ChatGPT auth profile"):
            resolve_chatgpt_auth_profile(
                litellm_params={"chatgpt_auth_profile": "missing-profile"}
            )

    def test_get_account_id_from_id_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_path = tmp_path / "auth.json"
        _write_auth_file(auth_path, {"id_token": id_token})

        authenticator = Authenticator()

        account_id = authenticator.get_account_id()

        assert account_id == "acct-123"
        saved_auth_data = json.loads(auth_path.read_text())
        assert saved_auth_data["account_id"] == "acct-123"

    def test_create_browser_login_session_uses_pkce_and_state(self, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", "/tmp/chatgpt-default")
        authenticator = Authenticator()

        session = authenticator.create_browser_login_session(
            allowed_workspace_id="workspace-123"
        )

        parsed = urlparse(session.authorize_url)
        params = parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert parsed.netloc == "auth.openai.com"
        assert parsed.path == "/oauth/authorize"
        assert params["response_type"] == ["code"]
        assert params["redirect_uri"] == [session.redirect_uri]
        assert params["state"] == [session.state]
        assert params["code_challenge_method"] == ["S256"]
        assert params["allowed_workspace_id"] == ["workspace-123"]
        assert len(session.code_verifier) > 40

    def test_complete_browser_login_exchanges_code_and_persists_auth(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        authenticator = Authenticator()
        session = authenticator.create_browser_login_session()

        with patch.object(
            authenticator,
            "_exchange_authorization_code_for_tokens",
            return_value={
                "access_token": _make_jwt({"exp": int(time.time()) + 3600}),
                "refresh_token": "refresh-123",
                "id_token": _make_jwt(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
                ),
            },
        ) as mock_exchange:
            tokens = authenticator.complete_browser_login(
                session,
                f"{session.redirect_uri}?code=auth-code-123&state={session.state}",
            )

        assert tokens["refresh_token"] == "refresh-123"
        mock_exchange.assert_called_once_with(
            authorization_code="auth-code-123",
            redirect_uri=session.redirect_uri,
            code_verifier=session.code_verifier,
        )
        saved_auth_data = json.loads(Path(authenticator.auth_file).read_text())
        assert saved_auth_data["refresh_token"] == "refresh-123"
        assert saved_auth_data["account_id"] == "acct-123"

    def test_complete_browser_login_rejects_state_mismatch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        authenticator = Authenticator()
        session = authenticator.create_browser_login_session()

        with pytest.raises(GetAccessTokenError, match="state mismatch"):
            authenticator.complete_browser_login(
                session,
                f"{session.redirect_uri}?code=auth-code-123&state=wrong-state",
            )

    def test_same_profile_refresh_is_locked(self, tmp_path):
        litellm.chatgpt_auth_profiles = {
            "account-a": {"token_dir": str(tmp_path / "account-a")}
        }
        authenticator = get_chatgpt_authenticator(
            {"chatgpt_auth_profile": "account-a"}
        )
        auth_path = Path(authenticator.auth_file)
        _write_auth_file(
            auth_path,
            {
                "access_token": "expired-token",
                "refresh_token": "refresh-123",
                "expires_at": time.time() - 10,
            },
        )

        refresh_call_count = 0
        refresh_count_lock = threading.Lock()

        def _refresh_tokens(refresh_token: str):
            nonlocal refresh_call_count
            assert refresh_token == "refresh-123"
            with refresh_count_lock:
                refresh_call_count += 1
            time.sleep(0.05)
            refreshed = {
                "access_token": _make_jwt({"exp": int(time.time()) + 3600}),
                "refresh_token": refresh_token,
                "id_token": _make_jwt({"exp": int(time.time()) + 3600}),
            }
            authenticator._write_auth_file(authenticator._build_auth_record(refreshed))
            return refreshed

        with patch.object(authenticator, "_refresh_tokens", side_effect=_refresh_tokens):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(authenticator.get_access_token) for _ in range(2)]
                results = [future.result() for future in futures]

        assert results[0] == results[1]
        assert refresh_call_count == 1
