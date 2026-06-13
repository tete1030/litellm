from __future__ import annotations

import json
import shutil
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import click

import litellm

from .authenticator import BrowserLoginSession, get_chatgpt_authenticator
from . import usage_service
from .inventory_tools import (
    doctor_inventory,
    load_inventory,
    render_config,
    report_as_text,
    update_rendered_config_file,
)

ChatGPTUsageMetrics = usage_service.ChatGPTUsageMetrics
ChatGPTRateLimitResetCreditsMetrics = usage_service.ChatGPTRateLimitResetCreditsMetrics
RateLimitResetCreditsResult = usage_service.RateLimitResetCreditsResult
UsageResult = usage_service.UsageResult
UsageWindow = usage_service.UsageWindow
_fetch_usage_for_profile = usage_service.fetch_usage_for_profile
_fetch_rate_limit_reset_credits_for_profile = (
    usage_service.fetch_rate_limit_reset_credits_for_profile
)
_consume_rate_limit_reset_credit_for_profile = (
    usage_service.consume_rate_limit_reset_credit_for_profile
)
_get_usage_url = usage_service.get_usage_url
_get_rate_limit_reset_credits_url = usage_service.get_rate_limit_reset_credits_url
_normalize_usage_payload = usage_service.normalize_usage_payload

DEFAULT_CONFIG_ENV_VAR = "CHATGPT_INVENTORY_PATH"
DEFAULT_CONFIG_DIR = Path.home() / ".config/litellm"
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / "inventory.yaml"
DEFAULT_CONFIG_CANDIDATES = (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_DIR / "inventory.yml",
)
DEFAULT_CONFIG_HELP = (
    "$CHATGPT_INVENTORY_PATH, ~/.config/litellm/inventory.yaml, ~/.config/litellm/inventory.yml"
)
DEFAULT_METRICS_PORT = 9464
DEFAULT_METRICS_INTERVAL_SECONDS = 300


def _iter_default_config_candidates() -> List[Path]:
    candidates: List[Path] = []
    env_config = litellm.get_secret(DEFAULT_CONFIG_ENV_VAR)
    if isinstance(env_config, str) and env_config.strip():
        candidates.append(Path(env_config.strip()).expanduser())
    candidates.extend(DEFAULT_CONFIG_CANDIDATES)
    return candidates


def _resolve_config_path(
    config_path: Optional[str], *, require_exists: bool = True
) -> Path:
    if config_path:
        resolved = Path(config_path).expanduser().resolve()
    else:
        resolved = next(
            (candidate.expanduser().resolve() for candidate in _iter_default_config_candidates() if candidate.exists()),
            DEFAULT_CONFIG.expanduser().resolve(),
        )

    if require_exists and not resolved.exists():
        raise click.ClickException(f"Inventory not found: {resolved}")
    return resolved


def _load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise click.ClickException(f"Inventory not found: {config_path}")

    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(config_path.read_text()) or {}

        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise click.ClickException(
                "YAML inventory support requires PyYAML. Install with `pip install 'litellm[proxy]'` "
                "or pass a JSON config file."
            ) from exc

        return yaml.safe_load(config_path.read_text()) or {}
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Failed reading inventory file {config_path}: {exc}"
        ) from exc


def _load_yaml_config_round_trip(config_path: Path) -> Dict[str, Any]:
    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "Editing YAML config files requires ruamel.yaml to preserve comments. "
            "Install with `pip install 'litellm[proxy]'` or use a JSON config file."
        ) from exc

    yaml = YAML()
    yaml.preserve_quotes = True

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.load(handle) or {}
    except Exception as exc:
        raise click.ClickException(
            f"Failed reading inventory file {config_path}: {exc}"
        ) from exc


def _load_chatgpt_auth_profiles(config_path: Path) -> Dict[str, Any]:
    data = _load_config_file(config_path)
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "profiles must be a mapping in the inventory file"
        )
    return profiles


def _load_config_with_format(config_path: Path) -> Tuple[Dict[str, Any], str]:
    suffix = config_path.suffix.lower()
    format_name = "json" if suffix == ".json" else "yaml"
    if not config_path.exists():
        return {}, format_name
    if format_name == "json":
        data = _load_config_file(config_path)
    else:
        data = _load_yaml_config_round_trip(config_path)
    return data, format_name


def _save_config_file(config_path: Path, data: Dict[str, Any], format_name: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if format_name == "json":
            config_path.write_text(json.dumps(data, indent=2) + "\n")
            return

        try:
            from ruamel.yaml import YAML
        except ModuleNotFoundError as exc:
            raise click.ClickException(
                "Editing YAML inventory files requires ruamel.yaml to preserve comments. "
                "Install with `pip install 'litellm[proxy]'` or use a JSON inventory file."
            ) from exc

        yaml = YAML()
        yaml.preserve_quotes = True
        buffer = StringIO()
        yaml.dump(data, buffer)
        config_path.write_text(buffer.getvalue(), encoding="utf-8")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Failed writing inventory file {config_path}: {exc}"
        ) from exc


def _default_token_dir_for_profile(
    profile: str, *, config_path: Optional[Path] = None
) -> str:
    if config_path is not None:
        return str((config_path.resolve().parent / "chatgpt" / profile).resolve())
    return str((Path.home() / ".config/litellm/chatgpt" / profile).resolve())


def _is_profile_enabled(profile_entry: Any) -> bool:
    if not isinstance(profile_entry, dict):
        return True
    return bool(profile_entry.get("enabled", True))


def _normalize_runtime_profiles(profiles: Dict[str, Any], *, enabled_only: bool) -> Dict[str, Any]:
    runtime_profiles: Dict[str, Any] = {}
    for name, entry in profiles.items():
        if enabled_only and not _is_profile_enabled(entry):
            continue
        normalized = entry if isinstance(entry, dict) else {}
        runtime_profiles[name] = {
            key: value
            for key, value in normalized.items()
            if key in {"token_dir", "auth_file"}
        }
    return runtime_profiles


def _profile_ls_row(profile: str, profile_entry: Any) -> Dict[str, Any]:
    entry = profile_entry if isinstance(profile_entry, dict) else {}
    return {
        "profile": profile,
        "token_dir": entry.get("token_dir", "-"),
        "auth_file": entry.get("auth_file", "auth.json (default)"),
        "enabled": _is_profile_enabled(entry),
        "weight": entry.get("weight", 1),
    }


def _update_profiles_in_config(
    config_path: Path,
    updater: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data, format_name = _load_config_with_format(config_path)
    profiles = data.get("profiles")
    if profiles is None:
        profiles = {}
        data["profiles"] = profiles
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "profiles must be a mapping in the inventory file"
        )
    data.setdefault("models", {})

    updater(profiles)
    _save_config_file(config_path, data, format_name)
    return data, profiles


def _update_models_in_config(
    config_path: Path,
    updater: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data, format_name = _load_config_with_format(config_path)
    models = data.get("models")
    if models is None:
        models = {}
        data["models"] = models
    if not isinstance(models, dict):
        raise click.ClickException(
            "models must be a mapping in the inventory file"
        )
    data.setdefault("profiles", {})

    updater(models)
    _save_config_file(config_path, data, format_name)
    return data, models


def _model_ls_row(model_name: str, model_entry: Any) -> Dict[str, Any]:
    entry = model_entry if isinstance(model_entry, dict) else {}
    return {
        "model_name": model_name,
        "provider_model": entry.get("provider_model", "-"),
        "mode": entry.get("mode", "responses"),
        "deployment_id_template": entry.get(
            "deployment_id_template", "chatgpt-{profile}"
        ),
        "enabled": bool(entry.get("enabled", True)),
    }


def _validate_weight(weight: int) -> int:
    if weight < 1:
        raise click.ClickException("weight must be >= 1")
    return weight


def _render_note() -> str:
    return "note: rerun `litellm-chatgpt render` to regenerate LiteLLM config.yaml."


def _prepare_profiles(config_path: Path, *, enabled_only: bool = False) -> Dict[str, Any]:
    profiles = _load_chatgpt_auth_profiles(config_path)
    litellm.chatgpt_auth_profiles = _normalize_runtime_profiles(
        profiles, enabled_only=enabled_only
    )
    return litellm.chatgpt_auth_profiles


def _load_auth_data(profile: str) -> tuple[Any, Path, Dict[str, Any]]:
    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_path = Path(authenticator.auth_file)
    auth_data = authenticator._read_auth_file() or {}
    return authenticator, auth_path, auth_data


def _get_login_session_path(auth_path: Path) -> Path:
    return auth_path.with_name("browser-login-session.json")


def _write_login_session(session_path: Path, session: BrowserLoginSession) -> None:
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(asdict(session), indent=2) + "\n")


def _read_login_session(session_path: Path) -> BrowserLoginSession:
    if not session_path.exists():
        raise click.ClickException(
            f"No pending browser login session found at {session_path}. Run `litellm-chatgpt login <profile>` first."
        )
    try:
        payload = json.loads(session_path.read_text())
        return BrowserLoginSession(**payload)
    except Exception as exc:
        raise click.ClickException(
            f"Failed reading browser login session {session_path}: {exc}"
        ) from exc


def _format_account(account_id: str) -> str:
    trimmed = account_id.strip()
    if len(trimmed) <= 12:
        return trimmed or "-"
    return f"{trimmed[:4]}...{trimmed[-4:]}"


def _format_credits(balance: Optional[float]) -> str:
    if balance is None:
        return "-"
    return f"${balance:.2f}"


def _format_reset_local(reset_at: Optional[int]) -> str:
    if not reset_at:
        return "-"
    return datetime.fromtimestamp(reset_at).astimezone().strftime("%Y-%m-%d %H:%M")


def _format_remaining(reset_at: Optional[int]) -> str:
    if not reset_at:
        return "-"
    remaining = int(reset_at - time.time())
    if remaining <= 0:
        return "0m"
    days, rem = divmod(remaining, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _fmt(row: Iterable[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    rendered = [_fmt(headers), _fmt(["-" * width for width in widths])]
    rendered.extend(_fmt(row) for row in rows)
    return "\n".join(rendered)


def _render_usage_report(results: List[UsageResult]) -> str:
    summary_rows = [
        [
            item.profile,
            item.plan if item.status == "ok" else "N/A",
            _format_account(item.account_id),
            _format_credits(item.credits_balance) if item.status == "ok" else "-",
        ]
        for item in results
    ]

    window_rows: List[List[str]] = []
    for item in results:
        if item.status != "ok":
            window_rows.append([item.profile, "-", "-", "-", item.error or "unknown error"])
            continue
        if not item.windows:
            window_rows.append([item.profile, "-", "-", "-", "-"])
            continue
        for index, window in enumerate(item.windows):
            window_rows.append(
                [
                    item.profile if index == 0 else "",
                    window.label,
                    f"{window.used_percent:.1f}%",
                    _format_reset_local(window.reset_at),
                    _format_remaining(window.reset_at),
                ]
            )

    return "\n\n".join(
        [
            _render_table(["profile", "plan", "account", "credits"], summary_rows),
            _render_table(
                ["profile", "window", "used", "reset(local)", "remaining"],
                window_rows,
            ),
        ]
    )


def _summarize_reset_credit(credit: Dict[str, Any]) -> str:
    for key in ("credit_id", "id", "name", "title"):
        value = credit.get(key)
        if value not in (None, ""):
            return f"{key}={value}"

    parts: List[str] = []
    for key in ("available", "status", "remaining_count", "redeem_request_id", "code"):
        value = credit.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")

    if parts:
        return ", ".join(parts)
    return json.dumps(credit, sort_keys=True, default=str)


def _render_rate_limit_reset_report(results: List[RateLimitResetCreditsResult]) -> str:
    summary_rows = [
        [
            item.profile,
            str(item.available_count) if item.status == "ok" and item.available_count is not None else "-",
            str(len(item.credits)) if item.status == "ok" else "-",
            item.status,
        ]
        for item in results
    ]

    credit_rows: List[List[str]] = []
    for item in results:
        if item.status != "ok":
            credit_rows.append([item.profile, item.error or "unknown error"])
            continue
        if not item.credits:
            credit_rows.append([item.profile, "-"])
            continue
        for index, credit in enumerate(item.credits):
            credit_rows.append(
                [
                    item.profile if index == 0 else "",
                    _summarize_reset_credit(credit),
                ]
            )

    return "\n\n".join(
        [
            _render_table(
                ["profile", "available", "credits", "status"],
                summary_rows,
            ),
            _render_table(["profile", "credit"], credit_rows),
        ]
    )


def _refresh_usage_metrics(
    config_path: Path, usage_url: str, metrics: ChatGPTUsageMetrics
) -> List[UsageResult]:
    profiles = _prepare_profiles(config_path, enabled_only=True)
    results = [_fetch_usage_for_profile(name, usage_url) for name in sorted(profiles.keys())]
    metrics.update(results)
    return results


def _refresh_rate_limit_reset_metrics(
    config_path: Path,
    reset_credits_url: str,
    metrics: ChatGPTRateLimitResetCreditsMetrics,
) -> List[RateLimitResetCreditsResult]:
    profiles = _prepare_profiles(config_path, enabled_only=True)
    results = [
        _fetch_rate_limit_reset_credits_for_profile(name, reset_credits_url)
        for name in sorted(profiles.keys())
    ]
    metrics.update(results)
    return results


@click.group(name="litellm-chatgpt")
def cli() -> None:
    """Manage ChatGPT OAuth profiles for LiteLLM."""


@cli.group(name="profile")
def profile_group() -> None:
    """Manage ChatGPT inventory profile entries."""


@profile_group.command(name="add")
@click.argument("profile")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--token-dir",
    help="Token directory for this profile. Defaults to <inventory dir>/chatgpt/<profile> when an inventory file is resolved, otherwise ~/.config/litellm/chatgpt/<profile>.",
)
@click.option(
    "--auth-file",
    help="Optional auth file path or file name for this profile entry.",
)
def profile_add(
    profile: str,
    config_path: str,
    token_dir: Optional[str],
    auth_file: Optional[str],
) -> None:
    """Add or update a named ChatGPT auth profile in inventory."""
    resolved_config = _resolve_config_path(config_path, require_exists=False)

    def _updater(profiles: Dict[str, Any]) -> None:
        existing_entry = profiles.get(profile)
        profile_entry: Dict[str, Any] = (
            dict(existing_entry) if isinstance(existing_entry, dict) else {}
        )
        resolved_token_dir = token_dir or profile_entry.get("token_dir") or _default_token_dir_for_profile(
            profile, config_path=resolved_config
        )
        profile_entry["token_dir"] = str(Path(resolved_token_dir).expanduser().resolve())
        if auth_file:
            auth_path = Path(auth_file).expanduser()
            profile_entry["auth_file"] = (
                str(auth_path.resolve()) if auth_path.is_absolute() else auth_file
            )
        profile_entry.setdefault("enabled", True)
        profile_entry.setdefault("weight", 1)
        profiles[profile] = profile_entry

    _, profiles = _update_profiles_in_config(resolved_config, _updater)
    added_profile = profiles[profile]
    click.echo(f"updated profile: {profile}")
    click.echo(f"inventory:       {resolved_config}")
    click.echo(f"token_dir:       {added_profile.get('token_dir')}")
    click.echo(f"auth_file:       {added_profile.get('auth_file', 'auth.json (default)')}")
    click.echo(f"enabled:         {added_profile.get('enabled', True)}")
    click.echo(f"weight:          {added_profile.get('weight', 1)}")
    click.echo(_render_note())


@profile_group.command(name="ls")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit profiles as JSON instead of a table.",
)
def profile_ls(config_path: str, json_output: bool) -> None:
    """List configured ChatGPT auth profiles from inventory."""
    resolved_config = _resolve_config_path(config_path)
    data = _load_config_file(resolved_config)
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "profiles must be a mapping in the inventory file"
        )

    rows = [_profile_ls_row(profile, profiles[profile]) for profile in sorted(profiles.keys())]

    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    table_rows = [
        [
            item["profile"],
            str(item["token_dir"]),
            str(item["auth_file"]),
            str(item["enabled"]),
            str(item["weight"]),
        ]
        for item in rows
    ]
    click.echo(
        _render_table(
            ["profile", "token_dir", "auth_file", "enabled", "weight"],
            table_rows,
        )
    )


@profile_group.command(name="rm")
@click.argument("profile")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--purge-files",
    is_flag=True,
    help="Also delete the profile auth.json and remove the profile directory if it becomes empty.",
)
def profile_rm(profile: str, config_path: str, purge_files: bool) -> None:
    """Remove a named ChatGPT auth profile from inventory."""
    resolved_config = _resolve_config_path(config_path)
    inventory_profiles = _load_chatgpt_auth_profiles(resolved_config)
    if profile not in inventory_profiles:
        raise click.ClickException(
            f"Profile '{profile}' not found in profiles"
        )

    _prepare_profiles(resolved_config)
    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_path = Path(authenticator.auth_file)
    session_path = _get_login_session_path(auth_path)
    token_dir_path = Path(authenticator.token_dir)

    def _updater(profiles: Dict[str, Any]) -> None:
        del profiles[profile]

    _update_profiles_in_config(resolved_config, _updater)

    removed_session = False
    if session_path.exists():
        session_path.unlink()
        removed_session = True

    removed_auth = False
    removed_token_dir = False
    if purge_files:
        if auth_path.exists():
            auth_path.unlink()
            removed_auth = True

        if token_dir_path.exists() and token_dir_path.is_dir():
            try:
                token_dir_path.rmdir()
                removed_token_dir = True
            except OSError:
                removed_token_dir = False

    click.echo(f"removed profile: {profile}")
    click.echo(f"inventory:       {resolved_config}")
    if removed_session:
        click.echo(f"removed session: {session_path}")
    if purge_files:
        if removed_auth:
            click.echo(f"removed auth:    {auth_path}")
        if removed_token_dir:
            click.echo(f"removed dir:     {token_dir_path}")
        elif token_dir_path.exists():
            click.echo(
                f"note: token dir not removed (not empty): {token_dir_path}"
            )
    else:
        click.echo(
            "note: auth.json is kept by default; use --purge-files to delete auth.json and attempt to remove the profile directory."
        )
    click.echo(_render_note())


@profile_group.command(name="enable")
@click.argument("profile")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
def profile_enable(profile: str, config_path: str) -> None:
    """Enable a named ChatGPT auth profile in inventory."""
    resolved_config = _resolve_config_path(config_path)

    def _updater(profiles: Dict[str, Any]) -> None:
        if profile not in profiles:
            raise click.ClickException(f"Profile '{profile}' not found in profiles")
        entry = profiles.get(profile)
        profile_entry = dict(entry) if isinstance(entry, dict) else {}
        profile_entry["enabled"] = True
        profiles[profile] = profile_entry

    _, profiles = _update_profiles_in_config(resolved_config, _updater)
    click.echo(f"enabled profile: {profile}")
    click.echo(f"inventory:       {resolved_config}")
    click.echo(f"enabled:         {profiles[profile].get('enabled', True)}")
    click.echo(f"weight:          {profiles[profile].get('weight', 1)}")
    click.echo(_render_note())


@profile_group.command(name="disable")
@click.argument("profile")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
def profile_disable(profile: str, config_path: str) -> None:
    """Disable a named ChatGPT auth profile in inventory."""
    resolved_config = _resolve_config_path(config_path)

    def _updater(profiles: Dict[str, Any]) -> None:
        if profile not in profiles:
            raise click.ClickException(f"Profile '{profile}' not found in profiles")
        entry = profiles.get(profile)
        profile_entry = dict(entry) if isinstance(entry, dict) else {}
        profile_entry["enabled"] = False
        profiles[profile] = profile_entry

    _, profiles = _update_profiles_in_config(resolved_config, _updater)
    click.echo(f"disabled profile: {profile}")
    click.echo(f"inventory:       {resolved_config}")
    click.echo(f"enabled:         {profiles[profile].get('enabled', True)}")
    click.echo(f"weight:          {profiles[profile].get('weight', 1)}")
    click.echo(_render_note())


@profile_group.command(name="set-weight")
@click.argument("profile")
@click.argument("weight", type=int)
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
def profile_set_weight(profile: str, weight: int, config_path: str) -> None:
    """Set profile weight in inventory."""
    resolved_config = _resolve_config_path(config_path)
    weight = _validate_weight(weight)

    def _updater(profiles: Dict[str, Any]) -> None:
        if profile not in profiles:
            raise click.ClickException(f"Profile '{profile}' not found in profiles")
        entry = profiles.get(profile)
        profile_entry = dict(entry) if isinstance(entry, dict) else {}
        profile_entry["weight"] = weight
        profiles[profile] = profile_entry

    _, profiles = _update_profiles_in_config(resolved_config, _updater)
    click.echo(f"updated profile weight: {profile}")
    click.echo(f"inventory:       {resolved_config}")
    click.echo(f"enabled:         {profiles[profile].get('enabled', True)}")
    click.echo(f"weight:          {profiles[profile].get('weight', 1)}")
    click.echo(_render_note())


@cli.group(name="model")
def model_group() -> None:
    """Manage ChatGPT inventory model entries."""


@model_group.command(name="add")
@click.argument("model_name")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing models.",
)
@click.option("--provider-model", required=True)
@click.option("--mode", default="responses", show_default=True)
@click.option(
    "--deployment-id-template",
    default="chatgpt-{profile}",
    show_default=True,
)
@click.option("--disabled", is_flag=True)
def model_add(
    model_name: str,
    config_path: str,
    provider_model: str,
    mode: str,
    deployment_id_template: str,
    disabled: bool,
) -> None:
    """Add or update a named model in inventory."""
    resolved_config = _resolve_config_path(config_path, require_exists=False)

    def _updater(models: Dict[str, Any]) -> None:
        entry = models.get(model_name)
        model_entry = dict(entry) if isinstance(entry, dict) else {}
        model_entry["provider_model"] = provider_model
        model_entry["mode"] = mode
        model_entry["deployment_id_template"] = deployment_id_template
        model_entry["enabled"] = not disabled
        models[model_name] = model_entry

    _, models = _update_models_in_config(resolved_config, _updater)
    click.echo(f"updated model:    {model_name}")
    click.echo(f"inventory:        {resolved_config}")
    click.echo(f"provider_model:   {models[model_name].get('provider_model')}")
    click.echo(f"mode:             {models[model_name].get('mode', 'responses')}")
    click.echo(
        f"deployment_id_template: {models[model_name].get('deployment_id_template', 'chatgpt-{{profile}}')}"
    )
    click.echo(f"enabled:          {bool(models[model_name].get('enabled', True))}")
    click.echo(_render_note())


@model_group.command(name="ls")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing models.",
)
@click.option("--json", "json_output", is_flag=True)
def model_ls(config_path: str, json_output: bool) -> None:
    """List configured inventory models."""
    resolved_config = _resolve_config_path(config_path)
    data = _load_config_file(resolved_config)
    models = data.get("models") or {}
    if not isinstance(models, dict):
        raise click.ClickException("models must be a mapping in the inventory file")

    rows = [_model_ls_row(model_name, models[model_name]) for model_name in sorted(models.keys())]
    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    table_rows = [
        [
            item["model_name"],
            str(item["provider_model"]),
            str(item["mode"]),
            str(item["deployment_id_template"]),
            str(item["enabled"]),
        ]
        for item in rows
    ]
    click.echo(
        _render_table(
            ["model_name", "provider_model", "mode", "deployment_id_template", "enabled"],
            table_rows,
        )
    )


@model_group.command(name="rm")
@click.argument("model_name")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing models.",
)
def model_rm(model_name: str, config_path: str) -> None:
    """Remove a named model from inventory."""
    resolved_config = _resolve_config_path(config_path)

    def _updater(models: Dict[str, Any]) -> None:
        if model_name not in models:
            raise click.ClickException(f"Model '{model_name}' not found in models")
        del models[model_name]

    _update_models_in_config(resolved_config, _updater)
    click.echo(f"removed model:    {model_name}")
    click.echo(f"inventory:        {resolved_config}")
    click.echo(_render_note())


@model_group.command(name="enable")
@click.argument("model_name")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing models.",
)
def model_enable(model_name: str, config_path: str) -> None:
    """Enable a named model in inventory."""
    resolved_config = _resolve_config_path(config_path)

    def _updater(models: Dict[str, Any]) -> None:
        if model_name not in models:
            raise click.ClickException(f"Model '{model_name}' not found in models")
        entry = models.get(model_name)
        model_entry = dict(entry) if isinstance(entry, dict) else {}
        model_entry["enabled"] = True
        models[model_name] = model_entry

    _, models = _update_models_in_config(resolved_config, _updater)
    click.echo(f"enabled model:    {model_name}")
    click.echo(f"inventory:        {resolved_config}")
    click.echo(f"enabled:          {bool(models[model_name].get('enabled', True))}")
    click.echo(_render_note())


@model_group.command(name="disable")
@click.argument("model_name")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing models.",
)
def model_disable(model_name: str, config_path: str) -> None:
    """Disable a named model in inventory."""
    resolved_config = _resolve_config_path(config_path)

    def _updater(models: Dict[str, Any]) -> None:
        if model_name not in models:
            raise click.ClickException(f"Model '{model_name}' not found in models")
        entry = models.get(model_name)
        model_entry = dict(entry) if isinstance(entry, dict) else {}
        model_entry["enabled"] = False
        models[model_name] = model_entry

    _, models = _update_models_in_config(resolved_config, _updater)
    click.echo(f"disabled model:   {model_name}")
    click.echo(f"inventory:        {resolved_config}")
    click.echo(f"enabled:          {bool(models[model_name].get('enabled', True))}")
    click.echo(_render_note())


@cli.command(name="render")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles and models.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    help="Path to LiteLLM runtime config file to update in place.",
)
def render_inventory(config_path: str, out_path: str) -> None:
    """Render managed runtime config keys from inventory."""
    resolved_config = _resolve_config_path(config_path)
    resolved_out = Path(out_path).expanduser().resolve()
    try:
        inventory = load_inventory(resolved_config)
        rendered = render_config(inventory)
        update_rendered_config_file(resolved_out, rendered)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"rendered managed keys in {resolved_out} from {resolved_config}")
    click.echo(
        f"profiles={len(rendered.get('chatgpt_auth_profiles', {}))} model_entries={len(rendered.get('model_list', []))}"
    )


@cli.command(name="doctor")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles and models.",
)
@click.option(
    "--chatgpt-dir",
    required=True,
    help="Path to the chatgpt auth directory to verify against inventory.",
)
def doctor(config_path: str, chatgpt_dir: str) -> None:
    """Check inventory drift against chatgpt auth directories."""
    resolved_config = _resolve_config_path(config_path)
    resolved_chatgpt_dir = Path(chatgpt_dir).expanduser().resolve()
    try:
        inventory = load_inventory(resolved_config)
        rendered = render_config(inventory)
        report = doctor_inventory(
            inventory,
            chatgpt_dir=resolved_chatgpt_dir,
            rendered_config=rendered,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(report_as_text(report))


@cli.command(name="login")
@click.argument("profile")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Backup and remove the current auth.json before logging in again.",
)
@click.option(
    "--browser/--device",
    "browser_login",
    default=True,
    show_default=True,
    help="Use browser OAuth login (default) or opt into device-code login.",
)
@click.option(
    "--callback-url",
    help="Optional full callback URL captured after browser login approval.",
)
@click.option(
    "--open-browser",
    is_flag=True,
    help="Attempt to open the authorize URL in your default browser automatically.",
)
@click.option(
    "--redirect-uri",
    help="Override the OAuth redirect URI used for browser login.",
)
@click.option(
    "--allowed-workspace-id",
    help="Optional workspace restriction passed to the browser authorize URL.",
)
def login(
    profile: str,
    config_path: str,
    force: bool,
    browser_login: bool,
    callback_url: Optional[str],
    open_browser: bool,
    redirect_uri: Optional[str],
    allowed_workspace_id: Optional[str],
) -> None:
    """Login or relogin a ChatGPT OAuth profile for LiteLLM."""
    resolved_config = _resolve_config_path(config_path)
    _prepare_profiles(resolved_config)

    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_path = Path(authenticator.auth_file)
    session_path = _get_login_session_path(auth_path)

    click.echo(f"profile: {profile}")
    click.echo(f"inventory: {resolved_config}")
    click.echo(f"auth:    {auth_path}")

    if force and auth_path.exists():
        backup_path = auth_path.with_name(
            f"auth.backup-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        shutil.copy2(auth_path, backup_path)
        auth_path.unlink()
        click.echo(f"backed up existing auth to: {backup_path}")

    if force and session_path.exists():
        session_path.unlink()
        click.echo(f"removed pending browser login session: {session_path}")

    if browser_login:
        if callback_url:
            login_session = _read_login_session(session_path)
            token = authenticator.complete_browser_login(login_session, callback_url)[
                "access_token"
            ]
            if session_path.exists():
                session_path.unlink()
            click.echo("browser login complete")
        else:
            login_session = authenticator.create_browser_login_session(
                redirect_uri=redirect_uri,
                allowed_workspace_id=allowed_workspace_id,
            )
            _write_login_session(session_path, login_session)
            click.echo("browser login url:")
            click.echo(login_session.authorize_url)
            click.echo(f"session: {session_path}")
            click.echo(
                "After approving access in your browser, run this command again with --callback-url '<full callback url>'."
            )
            if open_browser:
                opened = webbrowser.open(login_session.authorize_url)
                if not opened:
                    click.echo("warning: failed to open a browser automatically; open the URL manually.")
            return
    else:
        click.echo("using device-code login")
        token = authenticator._login_device_code()["access_token"]

    account_id = authenticator.get_account_id()

    auth_data: Dict[str, Any] = {}
    if auth_path.exists():
        auth_data = json.loads(auth_path.read_text())

    click.echo("login complete")
    click.echo(f"account_id: {account_id}")
    click.echo(f"expires_at: {auth_data.get('expires_at')}")
    click.echo(f"has_token:  {bool(token)}")


@cli.command(name="usage")
@click.argument("profile", required=False)
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit usage data as JSON instead of a table.",
)
def usage(profile: Optional[str], config_path: str, json_output: bool) -> None:
    """Query usage for one or all configured ChatGPT profiles."""
    resolved_config = _resolve_config_path(config_path)
    if profile:
        profiles = _prepare_profiles(resolved_config)
        if profile not in profiles:
            raise click.ClickException(f"Profile {profile!r} not found in inventory")
        profile_names = [profile]
    else:
        profiles = _prepare_profiles(resolved_config, enabled_only=True)
        profile_names = sorted(profiles.keys())
    usage_url = _get_usage_url()

    results = [_fetch_usage_for_profile(name, usage_url) for name in profile_names]
    if json_output:
        click.echo(json.dumps([asdict(item) for item in results], indent=2))
        return

    click.echo(_render_usage_report(results))


@cli.group(name="reset")
def reset_group() -> None:
    """Query and consume ChatGPT rate-limit reset credits."""


@reset_group.command(name="ls")
@click.argument("profile", required=False)
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit reset credits data as JSON instead of a table.",
)
def reset_ls(profile: Optional[str], config_path: str, json_output: bool) -> None:
    """Query available rate-limit reset credits for one or all configured profiles."""
    resolved_config = _resolve_config_path(config_path)
    if profile:
        profiles = _prepare_profiles(resolved_config)
        if profile not in profiles:
            raise click.ClickException(f"Profile {profile!r} not found in inventory")
        profile_names = [profile]
    else:
        profiles = _prepare_profiles(resolved_config, enabled_only=True)
        profile_names = sorted(profiles.keys())

    results = [
        _fetch_rate_limit_reset_credits_for_profile(
            name, _get_rate_limit_reset_credits_url()
        )
        for name in profile_names
    ]
    if json_output:
        click.echo(json.dumps([asdict(item) for item in results], indent=2))
        return

    click.echo(_render_rate_limit_reset_report(results))


@reset_group.command(name="consume")
@click.argument("profile")
@click.argument("credit_id")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--redeem-request-id",
    help="Optional request identifier. Defaults to a new UUID if omitted.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the reset consume response as JSON instead of a table.",
)
def reset_consume(
    profile: str,
    credit_id: str,
    config_path: str,
    redeem_request_id: Optional[str],
    json_output: bool,
) -> None:
    """Consume a ChatGPT rate-limit reset credit."""
    resolved_config = _resolve_config_path(config_path)
    profiles = _prepare_profiles(resolved_config)
    if profile not in profiles:
        raise click.ClickException(f"Profile {profile!r} not found in inventory")

    request_id = redeem_request_id or str(uuid4())
    result = _consume_rate_limit_reset_credit_for_profile(
        profile,
        credit_id,
        request_id,
        _get_rate_limit_reset_credits_url(),
    )

    if json_output:
        click.echo(json.dumps(asdict(result), indent=2))
    else:
        click.echo(f"profile:          {profile}")
        click.echo(f"credit_id:        {credit_id}")
        click.echo(f"redeem_request_id: {request_id}")
        click.echo(f"status:           {result.status}")
        if result.code:
            click.echo(f"code:             {result.code}")
        click.echo(_render_rate_limit_reset_report([result]))

    if result.status != "ok":
        raise click.ClickException(result.error or f"Failed to consume reset credit {credit_id!r}")


@cli.command(name="metrics")
@click.option(
    "--inventory",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="ChatGPT inventory file containing profiles.",
)
@click.option(
    "--listen-host",
    default="0.0.0.0",
    show_default=True,
    help="Listen address for the Prometheus metrics HTTP server.",
)
@click.option(
    "--port",
    default=DEFAULT_METRICS_PORT,
    show_default=True,
    type=int,
    help="Listen port for the Prometheus metrics HTTP server.",
)
@click.option(
    "--interval-seconds",
    default=DEFAULT_METRICS_INTERVAL_SECONDS,
    show_default=True,
    type=int,
    help="Refresh interval for ChatGPT usage and reset polling.",
)
def metrics(config_path: Optional[str], listen_host: str, port: int, interval_seconds: int) -> None:
    """Serve Prometheus metrics for ChatGPT account usage and reset credits."""
    if interval_seconds <= 0:
        raise click.ClickException("--interval-seconds must be greater than 0")

    resolved_config = _resolve_config_path(config_path)
    usage_url = _get_usage_url()
    reset_credits_url = usage_service.get_rate_limit_reset_credits_url()

    try:
        from prometheus_client import CollectorRegistry, start_http_server
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "Prometheus metrics require prometheus_client. Install with `pip install prometheus-client`."
        ) from exc

    registry = CollectorRegistry()
    usage_metrics = ChatGPTUsageMetrics(registry=registry)
    reset_credits_metrics = ChatGPTRateLimitResetCreditsMetrics(registry=registry)

    results = _refresh_usage_metrics(resolved_config, usage_url, usage_metrics)
    reset_results = _refresh_rate_limit_reset_metrics(
        resolved_config,
        reset_credits_url,
        reset_credits_metrics,
    )
    start_http_server(port, addr=listen_host, registry=registry)
    click.echo(
        f"serving ChatGPT usage and reset metrics on http://{listen_host}:{port}/metrics "
        f"for {len(results)} profile(s); reset profiles {len(reset_results)}; refresh interval {interval_seconds}s"
    )

    while True:
        try:
            time.sleep(interval_seconds)
            _refresh_usage_metrics(resolved_config, usage_url, usage_metrics)
            _refresh_rate_limit_reset_metrics(
                resolved_config,
                reset_credits_url,
                reset_credits_metrics,
            )
        except KeyboardInterrupt:
            click.echo("stopping ChatGPT usage and reset metrics exporter")
            return
        except Exception as exc:
            usage_metrics.mark_refresh_failure()
            reset_credits_metrics.mark_refresh_failure()
            click.echo(
                f"warning: failed refreshing ChatGPT usage and reset metrics: {exc}",
                err=True,
            )


if __name__ == "__main__":
    cli()
