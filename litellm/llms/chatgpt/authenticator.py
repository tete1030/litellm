import base64
import hashlib
import json
import os
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    CHATGPT_API_BASE,
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_DEVICE_VERIFY_URL,
    CHATGPT_OAUTH_AUTHORIZE_URL,
    CHATGPT_OAUTH_SCOPE,
    CHATGPT_OAUTH_TOKEN_URL,
    get_chatgpt_originator,
    ChatGPTAuthProfileError,
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAccessTokenError,
)

TOKEN_EXPIRY_SKEW_SECONDS = 60
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEVICE_CODE_COOLDOWN_SECONDS = 5 * 60
DEVICE_CODE_POLL_SLEEP_SECONDS = 5
DEFAULT_CHATGPT_PROFILE_NAME = "default"
DEFAULT_CHATGPT_TOKEN_DIR = os.path.expanduser("~/.config/litellm/chatgpt")
DEFAULT_CHATGPT_AUTH_FILE = "auth.json"
DEFAULT_BROWSER_LOGIN_PORT = 1455
CHATGPT_AUTH_PROFILES_ENV_VARS = (
    "CHATGPT_AUTH_PROFILES_JSON",
    "CHATGPT_AUTH_PROFILES",
)

_AUTHENTICATOR_CACHE: Dict[str, "Authenticator"] = {}
_AUTHENTICATOR_CACHE_LOCK = threading.Lock()
_PROFILE_LOCKS: Dict[str, threading.RLock] = {}
_PROFILE_LOCKS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResolvedChatGPTAuthProfile:
    profile_name: str
    token_dir: str
    auth_file: str
    cache_key: str


@dataclass(frozen=True)
class BrowserLoginSession:
    authorize_url: str
    redirect_uri: str
    state: str
    code_verifier: str


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _get_default_token_dir() -> str:
    return _normalize_path(os.getenv("CHATGPT_TOKEN_DIR", DEFAULT_CHATGPT_TOKEN_DIR))


def _get_default_auth_file_name() -> str:
    auth_file_name = os.getenv("CHATGPT_AUTH_FILE", DEFAULT_CHATGPT_AUTH_FILE)
    return auth_file_name or DEFAULT_CHATGPT_AUTH_FILE


def _get_profile_default_auth_file_name(profile_name: str) -> str:
    if profile_name == DEFAULT_CHATGPT_PROFILE_NAME:
        return _get_default_auth_file_name()
    return DEFAULT_CHATGPT_AUTH_FILE


def _normalize_litellm_params(litellm_params: Optional[Any]) -> Dict[str, Any]:
    if litellm_params is None:
        return {}
    if isinstance(litellm_params, dict):
        return dict(litellm_params)
    if hasattr(litellm_params, "model_dump"):
        try:
            return litellm_params.model_dump()
        except Exception:
            return {}
    if hasattr(litellm_params, "dict"):
        try:
            return litellm_params.dict()
        except Exception:
            return {}
    return {}


def get_chatgpt_auth_profile_name(litellm_params: Optional[Any]) -> str:
    params = _normalize_litellm_params(litellm_params)
    profile_name = params.get("chatgpt_auth_profile")
    if profile_name is None:
        return DEFAULT_CHATGPT_PROFILE_NAME
    profile_name = str(profile_name).strip()
    return profile_name or DEFAULT_CHATGPT_PROFILE_NAME


def _load_chatgpt_auth_profiles_from_env() -> Dict[str, Any]:
    raw_profiles: Optional[str] = None
    for env_var in CHATGPT_AUTH_PROFILES_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            raw_profiles = value
            break
    if not raw_profiles:
        return {}
    try:
        parsed_profiles = json.loads(raw_profiles)
    except json.JSONDecodeError as exc:
        raise ChatGPTAuthProfileError(
            status_code=400,
            message=f"Invalid ChatGPT auth profile registry in environment: {exc}",
        ) from exc
    if not isinstance(parsed_profiles, dict):
        raise ChatGPTAuthProfileError(
            status_code=400,
            message="ChatGPT auth profile registry must be a JSON object keyed by profile name.",
        )
    return parsed_profiles


def _get_chatgpt_auth_profile_registry() -> Dict[str, Any]:
    env_profiles = _load_chatgpt_auth_profiles_from_env()
    configured_profiles = getattr(litellm, "chatgpt_auth_profiles", {}) or {}
    if configured_profiles and not isinstance(configured_profiles, dict):
        raise ChatGPTAuthProfileError(
            status_code=400,
            message="litellm.chatgpt_auth_profiles must be a dictionary keyed by profile name.",
        )
    merged_profiles: Dict[str, Any] = {}
    merged_profiles.update(env_profiles)
    merged_profiles.update(configured_profiles)
    return normalize_chatgpt_auth_profiles(merged_profiles)


def _resolve_profile_definition(
    profile_name: str,
    profile_definition: Any,
) -> ResolvedChatGPTAuthProfile:
    if not isinstance(profile_definition, dict):
        raise ChatGPTAuthProfileError(
            status_code=400,
            message=(
                f"ChatGPT auth profile '{profile_name}' must map to an object with "
                "'token_dir' and/or 'auth_file'."
            ),
        )

    raw_token_dir = profile_definition.get("token_dir")
    raw_auth_file = profile_definition.get("auth_file")

    token_dir: Optional[str] = None
    if raw_token_dir is not None:
        token_dir = _normalize_path(str(raw_token_dir))

    auth_file: Optional[str] = None
    if raw_auth_file is not None:
        auth_file_str = str(raw_auth_file)
        if os.path.isabs(auth_file_str):
            auth_file = _normalize_path(auth_file_str)
        elif token_dir is not None:
            auth_file = _normalize_path(os.path.join(token_dir, auth_file_str))
        else:
            raise ChatGPTAuthProfileError(
                status_code=400,
                message=(
                    f"ChatGPT auth profile '{profile_name}' defines a relative auth_file "
                    "but no token_dir."
                ),
            )

    if token_dir is None and auth_file is not None:
        token_dir = os.path.dirname(auth_file)

    if token_dir is None:
        if profile_name == DEFAULT_CHATGPT_PROFILE_NAME:
            token_dir = _get_default_token_dir()
        else:
            raise ChatGPTAuthProfileError(
                status_code=400,
                message=(
                    f"ChatGPT auth profile '{profile_name}' must define 'token_dir' "
                    "or 'auth_file'."
                ),
            )

    if auth_file is None:
        auth_file = _normalize_path(
            os.path.join(token_dir, _get_profile_default_auth_file_name(profile_name))
        )

    return ResolvedChatGPTAuthProfile(
        profile_name=profile_name,
        token_dir=token_dir,
        auth_file=auth_file,
        cache_key=auth_file,
    )


def resolve_chatgpt_auth_profile(
    litellm_params: Optional[Any] = None,
    profile_name: Optional[str] = None,
) -> ResolvedChatGPTAuthProfile:
    requested_profile = (
        profile_name or get_chatgpt_auth_profile_name(litellm_params)
    ).strip()
    if not requested_profile:
        requested_profile = DEFAULT_CHATGPT_PROFILE_NAME

    profile_registry = _get_chatgpt_auth_profile_registry()
    if requested_profile in profile_registry:
        return _resolve_profile_definition(
            profile_name=requested_profile,
            profile_definition=profile_registry[requested_profile],
        )

    if requested_profile == DEFAULT_CHATGPT_PROFILE_NAME:
        return _resolve_profile_definition(
            profile_name=DEFAULT_CHATGPT_PROFILE_NAME,
            profile_definition={},
        )

    raise ChatGPTAuthProfileError(
        status_code=400,
        message=(
            f"Unknown ChatGPT auth profile '{requested_profile}'. Define it under "
            "'chatgpt_auth_profiles' or CHATGPT_AUTH_PROFILES_JSON."
        ),
    )


def normalize_chatgpt_auth_profiles(
    profile_registry: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    if not profile_registry:
        return {}
    if not isinstance(profile_registry, dict):
        raise ChatGPTAuthProfileError(
            status_code=400,
            message="chatgpt_auth_profiles must be a dictionary keyed by profile name.",
        )

    normalized_profiles: Dict[str, Dict[str, str]] = {}
    resolved_auth_files: Dict[str, str] = {}
    for profile_name, profile_definition in profile_registry.items():
        resolved = _resolve_profile_definition(str(profile_name), profile_definition)
        existing_profile = resolved_auth_files.get(resolved.auth_file)
        if existing_profile is not None:
            raise ChatGPTAuthProfileError(
                status_code=400,
                message=(
                    "ChatGPT auth profiles '{}' and '{}' resolve to the same auth_file '{}'. "
                    "Each profile must use an isolated auth file."
                ).format(existing_profile, resolved.profile_name, resolved.auth_file),
            )
        resolved_auth_files[resolved.auth_file] = resolved.profile_name
        normalized_profiles[resolved.profile_name] = {
            "token_dir": resolved.token_dir,
            "auth_file": resolved.auth_file,
        }

    if DEFAULT_CHATGPT_PROFILE_NAME not in normalized_profiles:
        default_profile = _resolve_profile_definition(
            DEFAULT_CHATGPT_PROFILE_NAME, {}
        )
        existing_profile = resolved_auth_files.get(default_profile.auth_file)
        if existing_profile is not None:
            raise ChatGPTAuthProfileError(
                status_code=400,
                message=(
                    "ChatGPT auth profile '{}' resolves to the same auth_file '{}' as the implicit "
                    "default profile. Each profile must use an isolated auth file."
                ).format(existing_profile, default_profile.auth_file),
            )
    return normalized_profiles


def _get_profile_lock(cache_key: str) -> threading.RLock:
    with _PROFILE_LOCKS_LOCK:
        lock = _PROFILE_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.RLock()
            _PROFILE_LOCKS[cache_key] = lock
        return lock


def get_chatgpt_authenticator(litellm_params: Optional[Any] = None) -> "Authenticator":
    resolved_profile = resolve_chatgpt_auth_profile(litellm_params=litellm_params)
    with _AUTHENTICATOR_CACHE_LOCK:
        authenticator = _AUTHENTICATOR_CACHE.get(resolved_profile.cache_key)
        if authenticator is None:
            authenticator = Authenticator(profile=resolved_profile)
            _AUTHENTICATOR_CACHE[resolved_profile.cache_key] = authenticator
        return authenticator


def reset_chatgpt_authenticator_cache() -> None:
    with _AUTHENTICATOR_CACHE_LOCK:
        _AUTHENTICATOR_CACHE.clear()
    with _PROFILE_LOCKS_LOCK:
        _PROFILE_LOCKS.clear()


class Authenticator:
    def __init__(
        self,
        profile_name: Optional[str] = None,
        profile: Optional[ResolvedChatGPTAuthProfile] = None,
    ) -> None:
        self.profile = profile or resolve_chatgpt_auth_profile(profile_name=profile_name)
        self.profile_name = self.profile.profile_name
        self.token_dir = self.profile.token_dir
        self.auth_file = self.profile.auth_file
        self._lock = _get_profile_lock(self.profile.cache_key)
        self._ensure_token_dir()

    def get_api_base(self) -> str:
        return (
            os.getenv("CHATGPT_API_BASE")
            or os.getenv("OPENAI_CHATGPT_API_BASE")
            or CHATGPT_API_BASE
        )

    def _get_valid_access_token_from_auth_data(
        self, auth_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        if not auth_data:
            return None
        access_token = auth_data.get("access_token")
        if access_token and not self._is_token_expired(auth_data, access_token):
            return access_token
        return None

    def get_access_token(self) -> str:
        auth_data = self._read_auth_file()
        access_token = self._get_valid_access_token_from_auth_data(auth_data)
        if access_token:
            return access_token

        with self._lock:
            auth_data = self._read_auth_file()
            access_token = self._get_valid_access_token_from_auth_data(auth_data)
            if access_token:
                return access_token

            refresh_token = auth_data.get("refresh_token") if auth_data else None
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning(
                        "ChatGPT refresh token failed for profile '%s', re-login required: %s",
                        self.profile_name,
                        exc,
                    )

            cooldown_remaining = self._get_device_code_cooldown_remaining(auth_data)

        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return token

        with self._lock:
            auth_data = self._read_auth_file()
            access_token = self._get_valid_access_token_from_auth_data(auth_data)
            if access_token:
                return access_token

            refresh_token = auth_data.get("refresh_token") if auth_data else None
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning(
                        "ChatGPT refresh token failed for profile '%s', re-login required: %s",
                        self.profile_name,
                        exc,
                    )

            tokens = self._login_device_code()
            return tokens["access_token"]

    def get_account_id(self) -> Optional[str]:
        auth_data = self._read_auth_file()
        if not auth_data:
            return None
        account_id = auth_data.get("account_id")
        if account_id:
            return account_id
        id_token = auth_data.get("id_token")
        access_token = auth_data.get("access_token")
        derived = self._extract_account_id(id_token or access_token)
        if derived:
            with self._lock:
                latest_auth_data = self._read_auth_file() or {}
                if latest_auth_data.get("account_id"):
                    return latest_auth_data["account_id"]
                latest_auth_data["account_id"] = derived
                self._write_auth_file(latest_auth_data)
        return derived

    def _ensure_token_dir(self) -> None:
        try:
            os.makedirs(self.token_dir, exist_ok=True)
        except OSError as exc:
            raise ChatGPTAuthProfileError(
                status_code=500,
                message=(
                    f"ChatGPT auth profile '{self.profile_name}' could not create token "
                    f"directory '{self.token_dir}': {exc}"
                ),
            ) from exc

    def _read_auth_file(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ChatGPTAuthProfileError(
                status_code=500,
                message=(
                    f"Invalid ChatGPT auth file for profile '{self.profile_name}' at "
                    f"'{self.auth_file}': {exc}"
                ),
            ) from exc
        except OSError as exc:
            raise ChatGPTAuthProfileError(
                status_code=500,
                message=(
                    f"Failed reading ChatGPT auth file for profile '{self.profile_name}' at "
                    f"'{self.auth_file}': {exc}"
                ),
            ) from exc

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        auth_dir = os.path.dirname(self.auth_file) or self.token_dir
        self._ensure_token_dir()
        try:
            os.makedirs(auth_dir, exist_ok=True)
        except OSError as exc:
            raise ChatGPTAuthProfileError(
                status_code=500,
                message=(
                    f"ChatGPT auth profile '{self.profile_name}' could not create auth "
                    f"directory '{auth_dir}': {exc}"
                ),
            ) from exc
        temp_file_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=auth_dir,
                prefix="auth-",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                json.dump(data, temp_file)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file_path = temp_file.name
            os.replace(temp_file_path, self.auth_file)
        except OSError as exc:
            raise ChatGPTAuthProfileError(
                status_code=500,
                message=(
                    f"Failed writing ChatGPT auth file for profile '{self.profile_name}' at "
                    f"'{self.auth_file}': {exc}"
                ),
            ) from exc
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

    def _is_token_expired(self, auth_data: Dict[str, Any], access_token: str) -> bool:
        expires_at = auth_data.get("expires_at")
        if expires_at is None:
            expires_at = self._get_expires_at(access_token)
            if expires_at:
                auth_data["expires_at"] = expires_at
                with self._lock:
                    latest_auth_data = self._read_auth_file() or auth_data
                    latest_auth_data["expires_at"] = expires_at
                    self._write_auth_file(latest_auth_data)
        if expires_at is None:
            return True
        return time.time() >= float(expires_at) - TOKEN_EXPIRY_SKEW_SECONDS

    def _get_expires_at(self, token: str) -> Optional[int]:
        claims = self._decode_jwt_claims(token)
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
        return None

    def _decode_jwt_claims(self, token: str) -> Dict[str, Any]:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64 = parts[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            return json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return {}

    def _extract_account_id(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        claims = self._decode_jwt_claims(token)
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            account_id = auth_claims.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id:
                return account_id
        return None

    def create_browser_login_session(
        self,
        redirect_uri: Optional[str] = None,
        allowed_workspace_id: Optional[str] = None,
    ) -> BrowserLoginSession:
        resolved_redirect_uri = (
            redirect_uri
            or f"http://localhost:{DEFAULT_BROWSER_LOGIN_PORT}/auth/callback"
        )
        state = _base64url_encode(os.urandom(32))
        code_verifier = _base64url_encode(os.urandom(64))
        code_challenge = _base64url_encode(
            hashlib.sha256(code_verifier.encode("utf-8")).digest()
        )
        query_params = {
            "response_type": "code",
            "client_id": CHATGPT_CLIENT_ID,
            "redirect_uri": resolved_redirect_uri,
            "scope": CHATGPT_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": get_chatgpt_originator(),
        }
        if allowed_workspace_id:
            query_params["allowed_workspace_id"] = allowed_workspace_id
        authorize_url = (
            f"{CHATGPT_OAUTH_AUTHORIZE_URL}?{urlencode(query_params, doseq=False)}"
        )
        return BrowserLoginSession(
            authorize_url=authorize_url,
            redirect_uri=resolved_redirect_uri,
            state=state,
            code_verifier=code_verifier,
        )

    def complete_browser_login(
        self,
        session: BrowserLoginSession,
        callback_url: str,
    ) -> Dict[str, str]:
        parsed = urlparse(callback_url.strip())
        callback_query = callback_url.strip()
        if parsed.scheme and parsed.netloc:
            callback_query = parsed.query
        elif callback_query.startswith("?"):
            callback_query = callback_query[1:]

        query_params = parse_qs(callback_query, keep_blank_values=True)
        error = query_params.get("error", [None])[0]
        if error:
            error_description = query_params.get("error_description", [None])[0]
            raise GetAccessTokenError(
                message=(
                    f"Browser login failed: {error}"
                    + (
                        f" ({error_description})"
                        if error_description
                        else ""
                    )
                ),
                status_code=400,
            )

        callback_state = query_params.get("state", [None])[0]
        if callback_state != session.state:
            raise GetAccessTokenError(
                message="Browser login failed: state mismatch in callback URL.",
                status_code=400,
            )

        code = query_params.get("code", [None])[0]
        if not code:
            raise GetAccessTokenError(
                message="Browser login failed: callback URL did not contain an authorization code.",
                status_code=400,
            )

        tokens = self._exchange_authorization_code_for_tokens(
            authorization_code=code,
            redirect_uri=session.redirect_uri,
            code_verifier=session.code_verifier,
        )
        auth_data = self._build_auth_record(tokens)
        self._write_auth_file(auth_data)
        return tokens

    def _login_device_code(self) -> Dict[str, str]:
        cooldown_remaining = self._get_device_code_cooldown_remaining(
            self._read_auth_file()
        )
        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return {"access_token": token}

        device_code = self._request_device_code()
        self._record_device_code_request()
        print(  # noqa: T201
            "Sign in with ChatGPT using device code:\n"
            f"1) Visit {CHATGPT_DEVICE_VERIFY_URL}\n"
            f"2) Enter code: {device_code['user_code']}\n"
            "Device codes are a common phishing target. Never share this code.",
            flush=True,
        )
        auth_code = self._poll_for_authorization_code(device_code)
        tokens = self._exchange_code_for_tokens(auth_code)
        auth_data = self._build_auth_record(tokens)
        self._write_auth_file(auth_data)
        return tokens

    def _request_device_code(self) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            resp = client.post(
                CHATGPT_DEVICE_CODE_URL,
                json={"client_id": CHATGPT_CLIENT_ID},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: {exc}",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: {exc}",
                status_code=400,
            ) from exc

        device_auth_id = data.get("device_auth_id")
        user_code = data.get("user_code") or data.get("usercode")
        interval = data.get("interval")
        if not device_auth_id or not user_code:
            raise GetDeviceCodeError(
                message=f"Device code response missing fields: {data}",
                status_code=400,
            )
        return {
            "device_auth_id": device_auth_id,
            "user_code": user_code,
            "interval": str(interval or "5"),
        }

    def _poll_for_authorization_code(
        self, device_code: Dict[str, str]
    ) -> Dict[str, str]:
        client = _get_httpx_client()
        interval = int(device_code.get("interval", "5"))
        start_time = time.time()
        while time.time() - start_time < DEVICE_CODE_TIMEOUT_SECONDS:
            try:
                resp = client.post(
                    CHATGPT_DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": device_code["device_auth_id"],
                        "user_code": device_code["user_code"],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if all(
                        key in data
                        for key in (
                            "authorization_code",
                            "code_challenge",
                            "code_verifier",
                        )
                    ):
                        return data
                if resp.status_code in (403, 404):
                    time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))
                    continue
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code in (403, 404):
                    time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))
                    continue
                raise GetAccessTokenError(
                    message=f"Polling failed: {exc}",
                    status_code=exc.response.status_code,
                ) from exc
            except Exception as exc:
                raise GetAccessTokenError(
                    message=f"Polling failed: {exc}",
                    status_code=400,
                ) from exc
            time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))

        raise GetAccessTokenError(
            message="Timed out waiting for device authorization",
            status_code=408,
        )

    def _exchange_authorization_code_for_tokens(
        self,
        authorization_code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            body = (
                "grant_type=authorization_code"
                f"&code={authorization_code}"
                f"&redirect_uri={redirect_uri}"
                f"&client_id={CHATGPT_CLIENT_ID}"
                f"&code_verifier={code_verifier}"
            )
            resp = client.post(
                CHATGPT_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=400,
            ) from exc

        if not all(
            key in data for key in ("access_token", "refresh_token", "id_token")
        ):
            raise GetAccessTokenError(
                message=f"Token exchange response missing fields: {data}",
                status_code=400,
            )
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "id_token": data["id_token"],
        }

    def _exchange_code_for_tokens(self, code_data: Dict[str, str]) -> Dict[str, str]:
        return self._exchange_authorization_code_for_tokens(
            authorization_code=code_data["authorization_code"],
            redirect_uri=f"{CHATGPT_AUTH_BASE}/deviceauth/callback",
            code_verifier=code_data["code_verifier"],
        )

    def _refresh_tokens(self, refresh_token: str) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            resp = client.post(
                CHATGPT_OAUTH_TOKEN_URL,
                json={
                    "client_id": CHATGPT_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=(
                    f"Refresh token failed for profile '{self.profile_name}': {exc}"
                ),
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=(
                    f"Refresh token failed for profile '{self.profile_name}': {exc}"
                ),
                status_code=400,
            ) from exc

        access_token = data.get("access_token")
        id_token = data.get("id_token")
        if not access_token or not id_token:
            raise RefreshAccessTokenError(
                message=f"Refresh response missing fields: {data}",
                status_code=400,
            )

        refreshed = {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token", refresh_token),
            "id_token": id_token,
        }
        auth_data = self._build_auth_record(refreshed)
        self._write_auth_file(auth_data)
        return refreshed

    def _build_auth_record(self, tokens: Dict[str, str]) -> Dict[str, Any]:
        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")
        expires_at = self._get_expires_at(access_token) if access_token else None
        account_id = self._extract_account_id(id_token or access_token)
        return {
            "access_token": access_token,
            "refresh_token": tokens.get("refresh_token"),
            "id_token": id_token,
            "expires_at": expires_at,
            "account_id": account_id,
        }

    def _get_device_code_cooldown_remaining(
        self, auth_data: Optional[Dict[str, Any]]
    ) -> float:
        if not auth_data:
            return 0.0
        requested_at = auth_data.get("device_code_requested_at")
        if not isinstance(requested_at, (int, float, str)):
            return 0.0
        try:
            requested_at = float(requested_at)
        except (TypeError, ValueError):
            return 0.0
        elapsed = time.time() - requested_at
        remaining = DEVICE_CODE_COOLDOWN_SECONDS - elapsed
        return max(0.0, remaining)

    def _record_device_code_request(self) -> None:
        auth_data = self._read_auth_file() or {}
        auth_data["device_code_requested_at"] = time.time()
        self._write_auth_file(auth_data)

    def _wait_for_access_token(self, timeout_seconds: float) -> Optional[str]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            auth_data = self._read_auth_file()
            access_token = self._get_valid_access_token_from_auth_data(auth_data)
            if access_token:
                return access_token
            sleep_for = min(
                DEVICE_CODE_POLL_SLEEP_SECONDS, max(0.0, deadline - time.time())
            )
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        return None
