from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import get_chatgpt_authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    merge_chatgpt_headers,
)
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = get_chatgpt_authenticator().get_api_base()
        dynamic_api_key = api_key or "chatgpt-oauth"
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        authenticator = get_chatgpt_authenticator(litellm_params)
        try:
            access_token = authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        protected_header_keys = {
            "Authorization",
            "ChatGPT-Account-Id",
            "session_id",
            "content-type",
            "accept",
        }
        return merge_chatgpt_headers(
            headers=validated_headers,
            default_headers=default_headers,
            protected_header_keys=protected_header_keys,
        )

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        optional_params.setdefault("stream", False)
        return optional_params
