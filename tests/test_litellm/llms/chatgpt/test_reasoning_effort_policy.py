from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.chatgpt import reasoning_effort_policy
from litellm.llms.chatgpt.reasoning_effort_metrics import (
    ChatGPTReasoningEffortMetrics,
)


def _litellm_params(levels):
    return {
        "metadata": {
            "user_api_key_alias": "vk-reasoning",
            "user_api_key_auth_metadata": {
                "chatgpt_reasoning_effort_policy": {
                    "version": 1,
                    "models": {
                        "gpt-5.6-sol": {
                            "levels": levels,
                        }
                    },
                }
            },
        }
    }


def test_replace_reasoning_effort_preserves_summary(monkeypatch):
    metrics = MagicMock()
    monkeypatch.setattr(
        reasoning_effort_policy,
        "get_chatgpt_reasoning_effort_metrics",
        lambda: metrics,
    )
    litellm_params = _litellm_params(
        {"max": {"action": "replace", "target": "xhigh"}}
    )

    request = reasoning_effort_policy.enforce_chatgpt_reasoning_effort_policy(
        request={"reasoning": {"effort": "max", "summary": "detailed"}},
        model="chatgpt/gpt-5.6-sol",
        litellm_params=litellm_params,
    )

    assert request["reasoning"] == {"effort": "xhigh", "summary": "detailed"}
    metadata = litellm_params["metadata"]
    assert metadata["chatgpt_requested_reasoning_effort"] == "max"
    assert metadata["chatgpt_effective_reasoning_effort"] == "xhigh"
    assert metadata["chatgpt_reasoning_effort_action"] == "replace"
    assert metadata["chatgpt_reasoning_effort_policy_applied"] is True
    metrics.observe_decision.assert_called_once_with(
        model="gpt-5.6-sol",
        virtual_key="vk-reasoning",
        requested_effort="max",
        effective_effort="xhigh",
        action="replace",
    )


def test_replace_reasoning_effort_with_default_removes_only_effort(monkeypatch):
    monkeypatch.setattr(
        reasoning_effort_policy,
        "get_chatgpt_reasoning_effort_metrics",
        MagicMock,
    )
    request = reasoning_effort_policy.enforce_chatgpt_reasoning_effort_policy(
        request={"reasoning": {"effort": "max", "summary": "detailed"}},
        model="gpt-5.6-sol",
        litellm_params=_litellm_params(
            {"max": {"action": "replace", "target": "default"}}
        ),
    )

    assert request["reasoning"] == {"summary": "detailed"}


def test_reject_reasoning_effort_records_decision_before_error(monkeypatch):
    metrics = MagicMock()
    monkeypatch.setattr(
        reasoning_effort_policy,
        "get_chatgpt_reasoning_effort_metrics",
        lambda: metrics,
    )
    litellm_params = _litellm_params({"ultra": {"action": "reject"}})

    with pytest.raises(UnsupportedParamsError, match="is not allowed"):
        reasoning_effort_policy.enforce_chatgpt_reasoning_effort_policy(
            request={"reasoning": {"effort": "ultra"}},
            model="gpt-5.6-sol",
            litellm_params=litellm_params,
        )

    metadata = litellm_params["metadata"]
    assert metadata["chatgpt_effective_reasoning_effort"] == "blocked"
    assert metadata["chatgpt_reasoning_effort_action"] == "reject"
    metrics.observe_decision.assert_called_once()


def test_request_metadata_cannot_spoof_virtual_key_policy(monkeypatch):
    monkeypatch.setattr(
        reasoning_effort_policy,
        "get_chatgpt_reasoning_effort_metrics",
        MagicMock,
    )
    litellm_params = {
        "metadata": {
            "chatgpt_reasoning_effort_policy": {
                "models": {
                    "gpt-5.6-sol": {
                        "levels": {"ultra": {"action": "reject"}}
                    }
                }
            }
        }
    }

    request = reasoning_effort_policy.enforce_chatgpt_reasoning_effort_policy(
        request={"reasoning": {"effort": "ultra"}},
        model="gpt-5.6-sol",
        litellm_params=litellm_params,
    )

    assert request["reasoning"]["effort"] == "ultra"
    assert litellm_params["metadata"]["chatgpt_reasoning_effort_action"] == "allow"


def test_reasoning_effort_metrics_export_decisions_and_usage():
    registry = CollectorRegistry()
    metrics = ChatGPTReasoningEffortMetrics(registry=registry)
    labels = {
        "model": "gpt-5.6-sol",
        "virtual_key": "vk-reasoning",
        "requested_effort": "max",
        "effective_effort": "xhigh",
        "action": "replace",
    }

    metrics.observe_decision(**labels)
    metrics.observe_usage(
        **labels,
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
        reasoning_tokens=5,
    )

    assert (
        registry.get_sample_value(
            "litellm_chatgpt_reasoning_effort_decisions_total", labels
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "litellm_chatgpt_reasoning_effort_reasoning_tokens_total", labels
        )
        == 5
    )
