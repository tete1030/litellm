from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import httpx

from litellm._logging import verbose_logger

from .authenticator import get_chatgpt_authenticator
from .common_utils import (
    get_chatgpt_codex_client_version,
    get_chatgpt_default_headers,
    get_chatgpt_originator,
    get_chatgpt_user_agent,
)
from .reasoning_effort_policy import (
    rejected_chatgpt_reasoning_efforts_for_model,
)

if TYPE_CHECKING:
    from litellm.router import Router


_CLIENT_VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9._-]+)?$"
)
_DEFAULT_CACHE_TTL_SECONDS = 300
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0
_DEFAULT_CACHE_PATH = "/tmp/litellm-chatgpt-model-catalog.json"


class ChatGPTModelCatalogError(Exception):
    pass


@dataclass
class _CatalogCacheEntry:
    fetched_at: float
    etag: Optional[str]
    models: List[Dict[str, Any]]


_catalog_cache: Dict[str, Dict[str, _CatalogCacheEntry]] = {}
_catalog_cache_loaded = False
_catalog_cache_lock = asyncio.Lock()


def normalize_codex_client_version(client_version: Optional[str]) -> str:
    resolved = client_version or get_chatgpt_codex_client_version()
    if not _CLIENT_VERSION_PATTERN.fullmatch(resolved):
        raise ValueError(f"Invalid Codex client version: {resolved!r}")
    return resolved


def get_chatgpt_profiles_for_models(
    model_ids: Sequence[str], llm_router: Optional["Router"]
) -> List[str]:
    if llm_router is None:
        return []

    profiles: List[str] = []
    seen = set()
    for model_id in model_ids:
        for deployment in llm_router.get_model_list(model_name=model_id) or []:
            params = deployment.get("litellm_params") or {}
            provider_model = params.get("model")
            profile = params.get("chatgpt_auth_profile")
            if not isinstance(provider_model, str) or not provider_model.startswith(
                "chatgpt/"
            ):
                continue
            if isinstance(profile, str) and profile and profile not in seen:
                seen.add(profile)
                profiles.append(profile)
    return profiles


def filter_chatgpt_model_catalog(
    models: Sequence[Dict[str, Any]],
    allowed_model_ids: Sequence[str],
    virtual_key_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    allowed = set(allowed_model_ids)
    filtered_models: List[Dict[str, Any]] = []
    for model in models:
        slug = model.get("slug")
        if slug not in allowed:
            continue
        copied_model = copy.deepcopy(model)
        rejected_efforts = rejected_chatgpt_reasoning_efforts_for_model(
            str(slug), virtual_key_metadata
        )
        reasoning_levels = copied_model.get("supported_reasoning_levels")
        if rejected_efforts and isinstance(reasoning_levels, list):
            copied_model["supported_reasoning_levels"] = [
                level
                for level in reasoning_levels
                if not isinstance(level, dict)
                or level.get("effort") not in rejected_efforts
            ]
        filtered_models.append(copied_model)
    return filtered_models


def _cache_path() -> Path:
    return Path(os.getenv("CHATGPT_MODEL_CATALOG_CACHE_PATH", _DEFAULT_CACHE_PATH))


def _cache_ttl_seconds() -> int:
    raw_value = os.getenv("CHATGPT_MODEL_CATALOG_CACHE_TTL_SECONDS")
    if raw_value is None:
        return _DEFAULT_CACHE_TTL_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return _DEFAULT_CACHE_TTL_SECONDS


def _request_timeout_seconds() -> float:
    raw_value = os.getenv("CHATGPT_MODEL_CATALOG_TIMEOUT_SECONDS")
    if raw_value is None:
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS


def _load_cache_from_disk() -> None:
    global _catalog_cache_loaded
    if _catalog_cache_loaded:
        return
    _catalog_cache_loaded = True

    path = _cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception as exc:
        verbose_logger.warning("Failed to load ChatGPT model catalog cache: %s", exc)
        return

    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, dict):
        return
    for client_version, profiles in versions.items():
        if not isinstance(client_version, str) or not isinstance(profiles, dict):
            continue
        for profile, raw_entry in profiles.items():
            if not isinstance(profile, str) or not isinstance(raw_entry, dict):
                continue
            models = raw_entry.get("models")
            if not isinstance(models, list) or not all(
                isinstance(model, dict) for model in models
            ):
                continue
            try:
                entry = _CatalogCacheEntry(
                    fetched_at=float(raw_entry.get("fetched_at", 0)),
                    etag=raw_entry.get("etag"),
                    models=[dict(model) for model in models],
                )
            except (TypeError, ValueError):
                continue
            _catalog_cache.setdefault(client_version, {})[profile] = entry


def _persist_cache_to_disk() -> None:
    path = _cache_path()
    payload = {
        "versions": {
            client_version: {
                profile: asdict(entry) for profile, entry in profiles.items()
            }
            for client_version, profiles in _catalog_cache.items()
        }
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        temporary_path.replace(path)
    except Exception as exc:
        verbose_logger.warning("Failed to persist ChatGPT model catalog cache: %s", exc)


def _entry_is_fresh(entry: _CatalogCacheEntry) -> bool:
    ttl = _cache_ttl_seconds()
    return ttl > 0 and time.time() - entry.fetched_at <= ttl


async def _fetch_profile_catalog(
    profile: str,
    client_version: str,
    cached_entry: Optional[_CatalogCacheEntry],
) -> _CatalogCacheEntry:
    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    originator = get_chatgpt_originator()
    headers = get_chatgpt_default_headers(
        authenticator.get_access_token(),
        authenticator.get_account_id(),
        user_agent=get_chatgpt_user_agent(originator, client_version),
    )
    headers["accept"] = "application/json"
    if cached_entry is not None and cached_entry.etag:
        headers["if-none-match"] = cached_entry.etag

    url = f"{authenticator.get_api_base().rstrip('/')}/models"
    async with httpx.AsyncClient(
        timeout=_request_timeout_seconds(), follow_redirects=True
    ) as client:
        response = await client.get(
            url,
            params={"client_version": client_version},
            headers=headers,
        )

    if response.status_code == 304 and cached_entry is not None:
        return _CatalogCacheEntry(
            fetched_at=time.time(),
            etag=cached_entry.etag,
            models=cached_entry.models,
        )

    response.raise_for_status()
    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not all(
        isinstance(model, dict) for model in models
    ):
        raise ChatGPTModelCatalogError(
            f"ChatGPT profile {profile!r} returned an invalid model catalog"
        )
    return _CatalogCacheEntry(
        fetched_at=time.time(),
        etag=response.headers.get("etag"),
        models=[dict(model) for model in models],
    )


async def get_chatgpt_model_catalog(
    client_version: Optional[str],
    profiles: Sequence[str],
    *,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    resolved_version = normalize_codex_client_version(client_version)
    unique_profiles = list(dict.fromkeys(profile for profile in profiles if profile))
    if not unique_profiles:
        raise ChatGPTModelCatalogError("No ChatGPT auth profiles are available")

    async with _catalog_cache_lock:
        _load_cache_from_disk()
        version_cache = _catalog_cache.setdefault(resolved_version, {})
        resolved_entries: Dict[str, _CatalogCacheEntry] = {}
        profiles_to_fetch: List[str] = []
        for profile in unique_profiles:
            cached_entry = version_cache.get(profile)
            if (
                not force_refresh
                and cached_entry is not None
                and _entry_is_fresh(cached_entry)
            ):
                resolved_entries[profile] = cached_entry
            else:
                profiles_to_fetch.append(profile)

        if profiles_to_fetch:
            results = await asyncio.gather(
                *[
                    _fetch_profile_catalog(
                        profile, resolved_version, version_cache.get(profile)
                    )
                    for profile in profiles_to_fetch
                ],
                return_exceptions=True,
            )
            cache_changed = False
            for profile, result in zip(profiles_to_fetch, results):
                if isinstance(result, Exception):
                    cached_entry = version_cache.get(profile)
                    if cached_entry is not None:
                        resolved_entries[profile] = cached_entry
                        verbose_logger.warning(
                            "ChatGPT model catalog refresh failed for profile %s; using cached catalog: %s",
                            profile,
                            result,
                        )
                    else:
                        verbose_logger.warning(
                            "ChatGPT model catalog refresh failed for profile %s: %s",
                            profile,
                            result,
                        )
                    continue
                version_cache[profile] = result
                resolved_entries[profile] = result
                cache_changed = True
            if cache_changed:
                _persist_cache_to_disk()

        if not resolved_entries:
            raise ChatGPTModelCatalogError(
                f"Unable to load ChatGPT model catalog for Codex {resolved_version}"
            )

        merged: List[Dict[str, Any]] = []
        models_by_slug: Dict[str, Dict[str, Any]] = {}
        for profile in unique_profiles:
            entry = resolved_entries.get(profile)
            if entry is None:
                continue
            for model in entry.models:
                slug = model.get("slug")
                if not isinstance(slug, str) or not slug:
                    continue
                existing = models_by_slug.get(slug)
                if existing is None:
                    copied = dict(model)
                    models_by_slug[slug] = copied
                    merged.append(copied)
                elif existing != model:
                    verbose_logger.warning(
                        "ChatGPT model metadata differs across profiles for %s; using profile %s",
                        slug,
                        unique_profiles[0],
                    )
        return merged


async def warm_chatgpt_model_catalog(
    profiles: Sequence[str], client_version: Optional[str] = None
) -> None:
    await get_chatgpt_model_catalog(
        client_version or get_chatgpt_codex_client_version(),
        profiles,
        force_refresh=True,
    )


def reset_chatgpt_model_catalog_cache() -> None:
    global _catalog_cache_loaded
    _catalog_cache.clear()
    _catalog_cache_loaded = False
