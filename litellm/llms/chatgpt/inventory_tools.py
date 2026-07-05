from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml as pyyaml

INVENTORY_ALLOWED_TOP_LEVEL_KEYS = {"defaults", "profiles", "models"}
MANAGED_RENDER_KEYS = ("chatgpt_auth_profiles", "model_list")


def _load_mapping_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".json":
        data = json.loads(text) or {}
    else:
        data = pyyaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Inventory root must be a mapping: {path}")
    return data


def _round_trip_yaml():
    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Updating YAML config files in place requires ruamel.yaml to preserve comments"
        ) from exc

    yaml = YAML()
    yaml.preserve_quotes = True
    return yaml


def load_inventory(path: Path) -> Dict[str, Any]:
    inventory = _load_mapping_file(path)
    unexpected_keys = sorted(
        key for key in inventory if key not in INVENTORY_ALLOWED_TOP_LEVEL_KEYS
    )
    if unexpected_keys:
        raise ValueError(
            "Unsupported top-level inventory keys: " + ", ".join(unexpected_keys)
        )
    defaults = inventory.setdefault("defaults", {})
    profiles = inventory.setdefault("profiles", {})
    models = inventory.setdefault("models", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a mapping")
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping")
    return inventory


def render_config(inventory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chatgpt_auth_profiles": {
            name: _profile_runtime_entry(entry if isinstance(entry, dict) else {})
            for name, entry in (inventory.get("profiles") or {}).items()
            if _profile_enabled(entry if isinstance(entry, dict) else {})
        },
        "model_list": _render_model_list(inventory),
    }


def update_rendered_config_file(
    path: Path,
    rendered: Dict[str, Any],
    *,
    managed_keys: Iterable[str] = MANAGED_RENDER_KEYS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".json":
        current = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(current, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        for key in managed_keys:
            if key in rendered:
                current[key] = rendered[key]
            elif key in current:
                del current[key]
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        return

    yaml = _round_trip_yaml()
    if path.exists():
        current = yaml.load(path.read_text()) or {}
        if not isinstance(current, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
    else:
        current = {}

    for key in managed_keys:
        if key in rendered:
            current[key] = rendered[key]
        elif key in current:
            del current[key]

    buffer = StringIO()
    yaml.dump(current, buffer)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def doctor_inventory(
    inventory: Dict[str, Any],
    *,
    chatgpt_dir: Path,
    rendered_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    profiles = inventory.get("profiles") or {}
    expected_auth_dirs = {
        _profile_auth_dir_name(name, entry if isinstance(entry, dict) else {}): name
        for name, entry in profiles.items()
    }
    auth_dirs = (
        {child.name for child in chatgpt_dir.iterdir() if child.is_dir()}
        if chatgpt_dir.exists()
        else set()
    )
    active_profiles = {
        name
        for name, entry in profiles.items()
        if _profile_enabled(entry if isinstance(entry, dict) else {})
    }
    missing_auth_dirs = sorted(
        name
        for auth_dir, name in expected_auth_dirs.items()
        if name in active_profiles and auth_dir not in auth_dirs
    )
    unmanaged_auth_dirs = sorted(
        auth_dir for auth_dir in auth_dirs if auth_dir not in expected_auth_dirs
    )
    disabled_profiles_with_auth = sorted(
        name
        for auth_dir, name in expected_auth_dirs.items()
        if name not in active_profiles and auth_dir in auth_dirs
    )
    report = {
        "active_profiles": sorted(active_profiles),
        "missing_auth_dirs": missing_auth_dirs,
        "unmanaged_auth_dirs": unmanaged_auth_dirs,
        "disabled_profiles_with_auth": disabled_profiles_with_auth,
    }
    if rendered_config is not None:
        report["rendered_model_count"] = len(rendered_config.get("model_list") or [])
    return report


def report_as_text(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _profile_enabled(profile_entry: Dict[str, Any]) -> bool:
    return bool(profile_entry.get("enabled", True))


def _model_enabled(model_entry: Dict[str, Any]) -> bool:
    return bool(model_entry.get("enabled", True))


def _profile_runtime_entry(profile_entry: Dict[str, Any]) -> Dict[str, Any]:
    runtime = {"token_dir": profile_entry["token_dir"]}
    auth_file = profile_entry.get("auth_file")
    if auth_file:
        runtime["auth_file"] = auth_file
    return runtime


def _profile_auth_dir_name(profile_name: str, profile_entry: Dict[str, Any]) -> str:
    auth_file = profile_entry.get("auth_file")
    if auth_file:
        return Path(str(auth_file)).parent.name or profile_name
    token_dir = profile_entry.get("token_dir")
    if token_dir:
        return Path(str(token_dir)).name or profile_name
    return profile_name


def _model_override(profile_entry: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    overrides = profile_entry.get("model_overrides") or {}
    if not isinstance(overrides, dict):
        return {}
    override = overrides.get(model_name) or {}
    return override if isinstance(override, dict) else {}


def _allow_fast_mode(
    inventory: Dict[str, Any], profile_entry: Dict[str, Any], override: Dict[str, Any]
) -> bool:
    defaults = inventory.get("defaults") or {}
    if "allow_fast_mode" in override:
        return bool(override.get("allow_fast_mode"))
    if "allow_fast_mode" in profile_entry:
        return bool(profile_entry.get("allow_fast_mode"))
    if "allow_fast_mode" in defaults:
        return bool(defaults.get("allow_fast_mode"))
    return True


def _deployment_id(
    profile_name: str, model_entry: Dict[str, Any], override: Dict[str, Any]
) -> str:
    deployment_id = override.get("deployment_id")
    if deployment_id:
        return str(deployment_id)
    template = model_entry.get("deployment_id_template")
    if template:
        return str(template).format(profile=profile_name)
    suffix = model_entry.get("id_suffix") or ""
    return f"chatgpt-{profile_name}{suffix}"


def _render_model_list(inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_list: List[Dict[str, Any]] = []
    profiles = inventory.get("profiles") or {}
    models = inventory.get("models") or {}
    for model_name, model_entry_any in models.items():
        model_entry = model_entry_any if isinstance(model_entry_any, dict) else {}
        if not _model_enabled(model_entry):
            continue
        provider_model = model_entry.get("provider_model")
        if not provider_model:
            raise ValueError(f"models.{model_name}.provider_model is required")
        for profile_name, profile_entry_any in profiles.items():
            profile_entry = profile_entry_any if isinstance(profile_entry_any, dict) else {}
            if not _profile_enabled(profile_entry):
                continue
            override = _model_override(profile_entry, model_name)
            if override.get("enabled") is False:
                continue
            mode = override.get("mode", model_entry.get("mode", "responses"))
            weight = override.get("weight", profile_entry.get("weight", 1))
            provider = override.get("provider_model", provider_model)
            allow_fast_mode = _allow_fast_mode(inventory, profile_entry, override)
            model_list.append(
                {
                    "model_name": model_name,
                    "model_info": {
                        "id": _deployment_id(profile_name, model_entry, override),
                        "mode": mode,
                    },
                    "litellm_params": {
                        "model": provider,
                        "chatgpt_auth_profile": profile_name,
                        "chatgpt_allow_fast_mode": allow_fast_mode,
                        "weight": weight,
                    },
                }
            )
    return model_list
