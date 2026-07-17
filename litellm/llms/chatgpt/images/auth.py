from dataclasses import dataclass
from typing import Any, Dict, Optional

from litellm.exceptions import AuthenticationError

from ..authenticator import get_chatgpt_authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    merge_chatgpt_headers,
)


@dataclass(frozen=True)
class ChatGPTImageAuth:
    api_base: str
    api_key: str
    headers: Dict[str, str]


def resolve_chatgpt_image_auth(
    litellm_params: Optional[Any],
    headers: Optional[dict] = None,
    model: str = "gpt-image-2",
) -> ChatGPTImageAuth:
    """Resolve deployment-scoped ChatGPT OAuth credentials for image requests."""
    authenticator = get_chatgpt_authenticator(litellm_params)
    try:
        access_token = authenticator.get_access_token()
    except GetAccessTokenError as exc:
        raise AuthenticationError(
            model=model,
            llm_provider="chatgpt",
            message=str(exc),
        ) from exc

    default_headers = get_chatgpt_default_headers(
        access_token=access_token,
        account_id=authenticator.get_account_id(),
        session_id=ensure_chatgpt_session_id(litellm_params),
    )
    default_headers["accept"] = "application/json"
    protected_header_keys = {
        "Authorization",
        "ChatGPT-Account-Id",
        "session_id",
        "content-type",
        "accept",
        "originator",
        "user-agent",
    }
    merged_headers = merge_chatgpt_headers(
        headers=headers or {},
        default_headers=default_headers,
        protected_header_keys=protected_header_keys,
    )

    return ChatGPTImageAuth(
        api_base=authenticator.get_api_base().rstrip("/"),
        api_key=access_token,
        headers=merged_headers,
    )
