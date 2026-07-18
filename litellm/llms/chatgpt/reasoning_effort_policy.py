from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from litellm.exceptions import UnsupportedParamsError

from .reasoning_effort_metrics import get_chatgpt_reasoning_effort_metrics


CHATGPT_REASONING_EFFORT_POLICY_KEY = "chatgpt_reasoning_effort_policy"
VALID_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
    "default",
}
VALID_REASONING_EFFORT_ACTIONS = {"allow", "replace", "reject"}


class ChatGPTReasoningEffortPolicyConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ChatGPTReasoningEffortDecision:
    requested_effort: str
    effective_effort: str
    action: str
    policy_source: str
    matched_model: Optional[str] = None


def normalize_chatgpt_reasoning_effort(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalize_chatgpt_reasoning_model(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    for prefix in ("chatgpt/", "responses/", "openai/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized or None


def _get_param_value(source: Optional[Any], key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    if source is None:
        return None
    return getattr(source, key, None)


def _iter_request_metadata(litellm_params: Optional[Any]) -> Iterable[Dict[str, Any]]:
    for key in ("metadata", "litellm_metadata"):
        candidate = _get_param_value(litellm_params, key)
        if isinstance(candidate, dict):
            yield candidate


def _iter_virtual_key_metadata(
    litellm_params: Optional[Any],
) -> Iterable[Dict[str, Any]]:
    for metadata in _iter_request_metadata(litellm_params):
        for nested_key in ("user_api_key_auth_metadata", "user_api_key_metadata"):
            nested = metadata.get(nested_key)
            if isinstance(nested, dict):
                yield nested
        auth_obj = metadata.get("user_api_key_auth")
        auth_metadata = getattr(auth_obj, "metadata", None)
        if isinstance(auth_metadata, dict):
            yield auth_metadata


def _get_virtual_key_metadata_with_policy(
    litellm_params: Optional[Any],
) -> Optional[Dict[str, Any]]:
    for metadata in _iter_virtual_key_metadata(litellm_params):
        if CHATGPT_REASONING_EFFORT_POLICY_KEY in metadata:
            return metadata
    return None


def _get_or_create_request_metadata(litellm_params: Optional[Any]) -> Dict[str, Any]:
    for metadata in _iter_request_metadata(litellm_params):
        return metadata
    if isinstance(litellm_params, dict):
        metadata: Dict[str, Any] = {}
        litellm_params["metadata"] = metadata
        return metadata
    if litellm_params is not None:
        metadata = {}
        setattr(litellm_params, "metadata", metadata)
        return metadata
    return {}


def _model_candidates(
    model: str, litellm_params: Optional[Any]
) -> Sequence[str]:
    candidates = []
    raw_candidates = [model, _get_param_value(litellm_params, "model")]
    for metadata in _iter_request_metadata(litellm_params):
        raw_candidates.extend([metadata.get("model_group"), metadata.get("model")])
    for raw_model in raw_candidates:
        normalized = normalize_chatgpt_reasoning_model(raw_model)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _validate_rule(rule: Any, model: str, effort: str) -> Tuple[str, Optional[str]]:
    if not isinstance(rule, dict):
        raise ChatGPTReasoningEffortPolicyConfigError(
            f"Policy rule for model={model!r}, effort={effort!r} must be an object"
        )
    action = normalize_chatgpt_reasoning_effort(rule.get("action"))
    if action not in VALID_REASONING_EFFORT_ACTIONS:
        raise ChatGPTReasoningEffortPolicyConfigError(
            f"Invalid reasoning effort action for model={model!r}, effort={effort!r}: {action!r}"
        )
    target = normalize_chatgpt_reasoning_effort(rule.get("target"))
    if action == "replace":
        if target not in VALID_REASONING_EFFORTS:
            raise ChatGPTReasoningEffortPolicyConfigError(
                f"Invalid replacement target for model={model!r}, effort={effort!r}: {target!r}"
            )
    elif target is not None:
        raise ChatGPTReasoningEffortPolicyConfigError(
            f"Reasoning effort action={action!r} cannot define a replacement target"
        )
    return action, target


def resolve_chatgpt_reasoning_effort_rule(
    *,
    model_candidates: Sequence[str],
    effort: str,
    virtual_key_metadata: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[str], Optional[str]]:
    if not virtual_key_metadata:
        return "allow", None, None
    policy = virtual_key_metadata.get(CHATGPT_REASONING_EFFORT_POLICY_KEY)
    if policy is None:
        return "allow", None, None
    if not isinstance(policy, dict):
        raise ChatGPTReasoningEffortPolicyConfigError(
            f"{CHATGPT_REASONING_EFFORT_POLICY_KEY} must be an object"
        )
    models = policy.get("models")
    if not isinstance(models, dict):
        raise ChatGPTReasoningEffortPolicyConfigError(
            f"{CHATGPT_REASONING_EFFORT_POLICY_KEY}.models must be an object"
        )

    for candidate in list(model_candidates) + ["*"]:
        model_policy = models.get(candidate)
        if model_policy is None:
            continue
        if not isinstance(model_policy, dict):
            raise ChatGPTReasoningEffortPolicyConfigError(
                f"Policy for model={candidate!r} must be an object"
            )
        levels = model_policy.get("levels")
        if not isinstance(levels, dict):
            raise ChatGPTReasoningEffortPolicyConfigError(
                f"Policy levels for model={candidate!r} must be an object"
            )
        rule = levels.get(effort)
        if rule is None:
            rule = levels.get("*")
        if rule is None:
            continue
        action, target = _validate_rule(rule, candidate, effort)
        return action, target, candidate
    return "allow", None, None


def get_chatgpt_reasoning_effort_from_request(request: Dict[str, Any]) -> str:
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict):
        normalized = normalize_chatgpt_reasoning_effort(reasoning.get("effort"))
        if normalized:
            return normalized
    reasoning_effort = request.get("reasoning_effort")
    if isinstance(reasoning_effort, dict):
        normalized = normalize_chatgpt_reasoning_effort(
            reasoning_effort.get("effort")
        )
    else:
        normalized = normalize_chatgpt_reasoning_effort(reasoning_effort)
    return normalized or "default"


def resolve_chatgpt_reasoning_effort_decision(
    request: Dict[str, Any], model: str, litellm_params: Optional[Any]
) -> ChatGPTReasoningEffortDecision:
    requested_effort = get_chatgpt_reasoning_effort_from_request(request)
    virtual_key_metadata = _get_virtual_key_metadata_with_policy(litellm_params)
    action, target, matched_model = resolve_chatgpt_reasoning_effort_rule(
        model_candidates=_model_candidates(model, litellm_params),
        effort=requested_effort,
        virtual_key_metadata=virtual_key_metadata,
    )
    if action == "reject":
        effective_effort = "blocked"
    elif action == "replace":
        effective_effort = target or "default"
    else:
        effective_effort = requested_effort
    return ChatGPTReasoningEffortDecision(
        requested_effort=requested_effort,
        effective_effort=effective_effort,
        action=action,
        policy_source="virtual_key" if matched_model else "default",
        matched_model=matched_model,
    )


def apply_chatgpt_reasoning_effort_decision(
    request: Dict[str, Any], decision: ChatGPTReasoningEffortDecision
) -> Dict[str, Any]:
    if decision.action != "replace":
        return request

    target = decision.effective_effort
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        updated_reasoning = dict(reasoning)
        if target == "default":
            updated_reasoning.pop("effort", None)
        else:
            updated_reasoning["effort"] = target
        if updated_reasoning:
            request["reasoning"] = updated_reasoning
        else:
            request.pop("reasoning", None)

    reasoning_effort = request.get("reasoning_effort")
    if isinstance(reasoning_effort, dict):
        updated_effort = dict(reasoning_effort)
        if target == "default":
            updated_effort.pop("effort", None)
        else:
            updated_effort["effort"] = target
        if updated_effort:
            request["reasoning_effort"] = updated_effort
        else:
            request.pop("reasoning_effort", None)
    elif "reasoning_effort" in request:
        if target == "default":
            request.pop("reasoning_effort", None)
        else:
            request["reasoning_effort"] = target
    return request


def _get_virtual_key_label(litellm_params: Optional[Any]) -> Optional[str]:
    for metadata in _iter_request_metadata(litellm_params):
        alias = metadata.get("user_api_key_alias")
        if alias is not None:
            return str(alias)
    return None


def _record_decision_metadata(
    litellm_params: Optional[Any], decision: ChatGPTReasoningEffortDecision
) -> None:
    metadata = _get_or_create_request_metadata(litellm_params)
    metadata.update(
        {
            "chatgpt_requested_reasoning_effort": decision.requested_effort,
            "chatgpt_effective_reasoning_effort": decision.effective_effort,
            "chatgpt_reasoning_effort_action": decision.action,
            "chatgpt_reasoning_effort_policy_source": decision.policy_source,
            "chatgpt_reasoning_effort_policy_applied": decision.policy_source
            != "default",
        }
    )


def enforce_chatgpt_reasoning_effort_policy(
    request: Dict[str, Any], model: str, litellm_params: Optional[Any]
) -> Dict[str, Any]:
    try:
        decision = resolve_chatgpt_reasoning_effort_decision(
            request=request, model=model, litellm_params=litellm_params
        )
    except ChatGPTReasoningEffortPolicyConfigError as exc:
        raise UnsupportedParamsError(
            message=f"Invalid ChatGPT reasoning effort policy: {exc}",
            llm_provider="chatgpt",
            model=model,
        ) from exc

    _record_decision_metadata(litellm_params, decision)
    get_chatgpt_reasoning_effort_metrics().observe_decision(
        model=normalize_chatgpt_reasoning_model(model) or model,
        virtual_key=_get_virtual_key_label(litellm_params),
        requested_effort=decision.requested_effort,
        effective_effort=decision.effective_effort,
        action=decision.action,
    )
    if decision.action == "reject":
        raise UnsupportedParamsError(
            message=(
                f"reasoning_effort={decision.requested_effort!r} is not allowed "
                f"for model={normalize_chatgpt_reasoning_model(model) or model!r} "
                "by the virtual key policy"
            ),
            llm_provider="chatgpt",
            model=model,
        )
    return apply_chatgpt_reasoning_effort_decision(request, decision)


def rejected_chatgpt_reasoning_efforts_for_model(
    model: str, virtual_key_metadata: Optional[Dict[str, Any]]
) -> set[str]:
    rejected = set()
    normalized_model = normalize_chatgpt_reasoning_model(model)
    if not normalized_model:
        return rejected
    for effort in VALID_REASONING_EFFORTS - {"default"}:
        action, _, _ = resolve_chatgpt_reasoning_effort_rule(
            model_candidates=[normalized_model],
            effort=effort,
            virtual_key_metadata=virtual_key_metadata,
        )
        if action == "reject":
            rejected.add(effort)
    return rejected
