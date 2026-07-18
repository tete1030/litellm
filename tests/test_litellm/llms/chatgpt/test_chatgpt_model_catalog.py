from unittest.mock import AsyncMock

import pytest

from litellm.llms.chatgpt import model_catalog


@pytest.fixture(autouse=True)
def reset_catalog_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "CHATGPT_MODEL_CATALOG_CACHE_PATH", str(tmp_path / "models-cache.json")
    )
    monkeypatch.setenv("CHATGPT_MODEL_CATALOG_CACHE_TTL_SECONDS", "300")
    model_catalog.reset_chatgpt_model_catalog_cache()
    yield
    model_catalog.reset_chatgpt_model_catalog_cache()


def test_filter_chatgpt_model_catalog_uses_exact_slugs():
    models = [
        {"slug": "gpt-5.6-sol"},
        {"slug": "gpt-5.3-codex-spark"},
    ]

    filtered = model_catalog.filter_chatgpt_model_catalog(
        models, ["gpt-5.6-sol", "gpt-5.3-codex"]
    )

    assert filtered == [{"slug": "gpt-5.6-sol"}]


def test_filter_catalog_keeps_replaced_effort_and_hides_rejected_effort():
    models = [
        {
            "slug": "gpt-5.6-sol",
            "supported_reasoning_levels": [
                {"effort": "xhigh"},
                {"effort": "max"},
                {"effort": "ultra"},
            ],
        }
    ]
    key_metadata = {
        "chatgpt_reasoning_effort_policy": {
            "models": {
                "gpt-5.6-sol": {
                    "levels": {
                        "max": {"action": "replace", "target": "xhigh"},
                        "ultra": {"action": "reject"},
                    }
                }
            }
        }
    }

    filtered = model_catalog.filter_chatgpt_model_catalog(
        models, ["gpt-5.6-sol"], virtual_key_metadata=key_metadata
    )

    assert [
        level["effort"] for level in filtered[0]["supported_reasoning_levels"]
    ] == ["xhigh", "max"]
    assert [
        level["effort"] for level in models[0]["supported_reasoning_levels"]
    ] == ["xhigh", "max", "ultra"]


@pytest.mark.asyncio
async def test_catalog_fetches_profiles_once_and_preserves_upstream_metadata(
    monkeypatch,
):
    fetch = AsyncMock(
        side_effect=[
            model_catalog._CatalogCacheEntry(
                fetched_at=1000,
                etag="etag-a",
                models=[
                    {
                        "slug": "gpt-5.6-sol",
                        "tool_mode": "code_mode_only",
                        "multi_agent_version": "v2",
                        "use_responses_lite": True,
                    }
                ],
            ),
            model_catalog._CatalogCacheEntry(
                fetched_at=1000,
                etag="etag-b",
                models=[
                    {
                        "slug": "gpt-5.6-luna",
                        "multi_agent_version": "v1",
                        "supported_reasoning_levels": [
                            {"effort": "max", "description": "Maximum reasoning"}
                        ],
                    }
                ],
            ),
        ]
    )
    monkeypatch.setattr(model_catalog, "_fetch_profile_catalog", fetch)
    monkeypatch.setattr(model_catalog.time, "time", lambda: 1000)

    models = await model_catalog.get_chatgpt_model_catalog(
        "0.144.1", ["profile-a", "profile-b"]
    )
    cached_models = await model_catalog.get_chatgpt_model_catalog(
        "0.144.1", ["profile-a", "profile-b"]
    )

    assert models == cached_models
    assert models[0]["tool_mode"] == "code_mode_only"
    assert models[0]["multi_agent_version"] == "v2"
    assert models[1]["multi_agent_version"] == "v1"
    assert models[1]["supported_reasoning_levels"] == [
        {"effort": "max", "description": "Maximum reasoning"}
    ]
    assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_catalog_uses_cached_profile_after_refresh_failure(monkeypatch):
    successful_fetch = AsyncMock(
        return_value=model_catalog._CatalogCacheEntry(
            fetched_at=1000,
            etag="etag-a",
            models=[{"slug": "gpt-5.5", "use_responses_lite": False}],
        )
    )
    monkeypatch.setattr(model_catalog, "_fetch_profile_catalog", successful_fetch)
    monkeypatch.setattr(model_catalog.time, "time", lambda: 1000)
    await model_catalog.get_chatgpt_model_catalog("0.144.1", ["profile-a"])

    monkeypatch.setattr(
        model_catalog,
        "_fetch_profile_catalog",
        AsyncMock(side_effect=RuntimeError("upstream unavailable")),
    )
    models = await model_catalog.get_chatgpt_model_catalog(
        "0.144.1", ["profile-a"], force_refresh=True
    )

    assert models == [{"slug": "gpt-5.5", "use_responses_lite": False}]


def test_normalize_codex_client_version_rejects_invalid_values():
    with pytest.raises(ValueError):
        model_catalog.normalize_codex_client_version("0.144.1\r\ninvalid")
