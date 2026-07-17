import base64
import mimetypes
import os
from typing import Any, Dict, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse


class ChatGPTImageError(BaseLLMException):
    pass


class ChatGPTImageEditConfig(BaseImageEditConfig):
    """Translate OpenAI image-edit inputs to the ChatGPT Codex JSON contract."""

    def get_supported_openai_params(self, model: str) -> list:
        return ["background", "n", "quality", "size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        supported_params = self.get_supported_openai_params(model)
        return {key: value for key, value in image_edit_optional_params.items() if key in supported_params}

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if not api_key:
            raise ValueError("ChatGPT OAuth access token is required")
        headers["Authorization"] = f"Bearer {api_key}"
        headers.setdefault("content-type", "application/json")
        headers.setdefault("accept", "application/json")
        return headers

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if not api_base:
            raise ValueError("ChatGPT API base is required")
        api_base = api_base.rstrip("/")
        if api_base.endswith("/images/edits"):
            return api_base
        return f"{api_base}/images/edits"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        images = image if isinstance(image, list) else [image]
        encoded_images = [self._to_image_url(image_item) for image_item in images if image_item is not None]
        if not encoded_images:
            raise ValueError("At least one image is required for ChatGPT image edits")

        request_body: Dict[str, Any] = {
            "model": model,
            "images": encoded_images,
        }
        if prompt is not None:
            request_body["prompt"] = prompt
        request_body.update(image_edit_optional_request_params)

        return request_body, cast(RequestFiles, [])

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        try:
            response_json = raw_response.json()
        except Exception as exc:
            raise ChatGPTImageError(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
                response=raw_response,
            ) from exc
        return ImageResponse(**response_json)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return ChatGPTImageError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @classmethod
    def _to_image_url(cls, image: Any) -> Dict[str, str]:
        if isinstance(image, dict):
            image_url = image.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise ValueError("ChatGPT image dictionaries require an image_url")
            return {"image_url": image_url}

        if isinstance(image, str) and image.startswith("data:"):
            return {"image_url": image}

        content_type: Optional[str] = None
        image_value = image
        if isinstance(image, tuple):
            if len(image) < 2:
                raise ValueError("Invalid image tuple")
            image_value = image[1]
            if len(image) >= 3 and isinstance(image[2], str):
                content_type = image[2]
            if content_type is None and isinstance(image[0], str):
                content_type = mimetypes.guess_type(image[0])[0]

        image_bytes = cls._read_image_bytes(image_value)
        if content_type is None:
            if isinstance(image_value, (str, os.PathLike)):
                content_type = mimetypes.guess_type(os.fspath(image_value))[0]
            content_type = content_type or ImageEditRequestUtils.get_image_content_type(image_value)
        payload = base64.b64encode(image_bytes).decode("ascii")
        return {"image_url": f"data:{content_type};base64,{payload}"}

    @staticmethod
    def _read_image_bytes(image: Any) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, (str, os.PathLike)):
            with open(os.fspath(image), "rb") as image_file:
                return image_file.read()
        if hasattr(image, "read"):
            current_position = image.tell() if hasattr(image, "tell") else None
            if hasattr(image, "seek"):
                image.seek(0)
            data = image.read()
            if current_position is not None and hasattr(image, "seek"):
                image.seek(current_position)
            if isinstance(data, bytes):
                return data
        raise ValueError("Unsupported image type for ChatGPT image edit")
