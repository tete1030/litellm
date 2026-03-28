from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

import litellm
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .authenticator import get_chatgpt_authenticator

DEFAULT_CONFIG = Path.home() / ".config/litellm/config.chatgpt-multi.yaml"
DEFAULT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


@dataclass
class UsageWindow:
    label: str
    used_percent: float
    reset_at: Optional[int]


@dataclass
class UsageResult:
    profile: str
    account_id: str
    plan: str
    credits_balance: Optional[float]
    windows: List[UsageWindow]
    status: str
    error: Optional[str] = None


def _load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise click.ClickException(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(config_path.read_text()) or {}

        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise click.ClickException(
                "YAML config support requires PyYAML. Install with `pip install 'litellm[proxy]'` "
                "or pass a JSON config file."
            ) from exc

        return yaml.safe_load(config_path.read_text()) or {}
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Failed reading config file {config_path}: {exc}"
        ) from exc


def _load_chatgpt_auth_profiles(config_path: Path) -> Dict[str, Any]:
    data = _load_config_file(config_path)
    profiles = data.get("chatgpt_auth_profiles") or {}
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "chatgpt_auth_profiles must be a mapping in the config file"
        )
    return profiles


def _load_config_with_format(config_path: Path) -> Tuple[Dict[str, Any], str]:
    data = _load_config_file(config_path)
    suffix = config_path.suffix.lower()
    format_name = "json" if suffix == ".json" else "yaml"
    return data, format_name


def _save_config_file(config_path: Path, data: Dict[str, Any], format_name: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if format_name == "json":
            config_path.write_text(json.dumps(data, indent=2) + "\n")
            return

        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise click.ClickException(
                "YAML config support requires PyYAML. Install with `pip install 'litellm[proxy]'` "
                "or use a JSON config file."
            ) from exc

        config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Failed writing config file {config_path}: {exc}"
        ) from exc


def _default_token_dir_for_profile(profile: str) -> str:
    return str((Path.home() / ".config/litellm/chatgpt" / profile).resolve())


def _ensure_model_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_list = data.get("model_list")
    if model_list is None:
        model_list = []
        data["model_list"] = model_list
    if not isinstance(model_list, list):
        raise click.ClickException("model_list must be a list in the config file")
    return model_list


def _deployment_matches_profile(deployment: Any, profile: str) -> bool:
    if not isinstance(deployment, dict):
        return False
    litellm_params = deployment.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return False
    return litellm_params.get("chatgpt_auth_profile") == profile


def _get_profile_deployments(data: Dict[str, Any], profile: str) -> List[Dict[str, Any]]:
    model_list = data.get("model_list") or []
    if not isinstance(model_list, list):
        return []
    return [item for item in model_list if _deployment_matches_profile(item, profile)]


def _format_profile_deployments(deployments: List[Dict[str, Any]]) -> str:
    if not deployments:
        return "-"
    rendered: List[str] = []
    for deployment in deployments:
        model_name = str(deployment.get("model_name") or "-")
        model_info = deployment.get("model_info") or {}
        deployment_id = "-"
        if isinstance(model_info, dict):
            deployment_id = str(model_info.get("id") or "-")
        rendered.append(f"{deployment_id} ({model_name})")
    return ", ".join(rendered)


def _update_profiles_in_config(
    config_path: Path,
    updater: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data, format_name = _load_config_with_format(config_path)
    profiles = data.get("chatgpt_auth_profiles")
    if profiles is None:
        profiles = {}
        data["chatgpt_auth_profiles"] = profiles
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "chatgpt_auth_profiles must be a mapping in the config file"
        )

    updater(profiles)
    _save_config_file(config_path, data, format_name)
    return data, profiles


def _prepare_profiles(config_path: Path) -> Dict[str, Any]:
    profiles = _load_chatgpt_auth_profiles(config_path)
    litellm.chatgpt_auth_profiles = profiles
    return profiles


def _load_auth_data(profile: str) -> tuple[Any, Path, Dict[str, Any]]:
    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_path = Path(authenticator.auth_file)
    auth_data = authenticator._read_auth_file() or {}
    return authenticator, auth_path, auth_data


def _get_usage_access_token(authenticator: Any, auth_data: Dict[str, Any]) -> str:
    access_token = auth_data.get("access_token")
    if access_token and not authenticator._is_token_expired(auth_data, access_token):
        return access_token

    refresh_token = auth_data.get("refresh_token")
    if refresh_token:
        refreshed = authenticator._refresh_tokens(refresh_token)
        return refreshed["access_token"]

    raise click.ClickException(
        f"Profile '{authenticator.profile_name}' is not logged in. Run `litellm-chatgpt login {authenticator.profile_name}`."
    )


def _normalize_usage_window(label: str, window_payload: Dict[str, Any]) -> UsageWindow:
    reset_at_raw = window_payload.get("reset_at")
    reset_at: Optional[int] = None
    if isinstance(reset_at_raw, (int, float)):
        reset_at = int(reset_at_raw)
    elif isinstance(reset_at_raw, str) and reset_at_raw.isdigit():
        reset_at = int(reset_at_raw)

    used_percent_raw = window_payload.get("used_percent", 0)
    try:
        used_percent = float(used_percent_raw)
    except (TypeError, ValueError):
        used_percent = 0.0

    return UsageWindow(
        label=label,
        used_percent=max(0.0, min(100.0, used_percent)),
        reset_at=reset_at,
    )


def _normalize_usage_payload(profile: str, account_id: str, payload: Dict[str, Any]) -> UsageResult:
    plan = str(payload.get("plan_type") or "unknown")
    credits_balance = None
    credits = payload.get("credits")
    if isinstance(credits, dict) and credits.get("balance") is not None:
        try:
            credits_balance = float(credits["balance"])
        except (TypeError, ValueError):
            credits_balance = None

    windows: List[UsageWindow] = []
    rate_limit = payload.get("rate_limit")
    if isinstance(rate_limit, dict):
        primary = rate_limit.get("primary_window")
        if isinstance(primary, dict):
            hours = int(primary.get("limit_window_seconds", 0) or 0) // 3600
            windows.append(
                _normalize_usage_window(f"{hours if hours > 0 else 3}h", primary)
            )
        secondary = rate_limit.get("secondary_window")
        if isinstance(secondary, dict):
            hours = int(secondary.get("limit_window_seconds", 0) or 0) // 3600
            label = "Day" if hours >= 24 else f"{hours if hours > 0 else 24}h"
            windows.append(_normalize_usage_window(label, secondary))

    return UsageResult(
        profile=profile,
        account_id=account_id,
        plan=plan,
        credits_balance=credits_balance,
        windows=windows,
        status="ok",
    )


def _fetch_usage_for_profile(profile: str, usage_url: str) -> UsageResult:
    authenticator, _, auth_data = _load_auth_data(profile)
    account_id = auth_data.get("account_id") or authenticator.get_account_id() or ""
    access_token = _get_usage_access_token(authenticator, auth_data)

    client = _get_httpx_client()
    response = client.get(
        usage_url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "litellm-chatgpt",
            "Accept": "application/json",
            **(
                {"ChatGPT-Account-Id": account_id}
                if account_id
                else {}
            ),
        },
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        body_text = response.text.strip() if response.text else str(exc)
        return UsageResult(
            profile=profile,
            account_id=account_id,
            plan="N/A",
            credits_balance=None,
            windows=[],
            status="error",
            error=f"usage request failed ({response.status_code}): {body_text}",
        )

    payload = response.json()
    return _normalize_usage_payload(profile, account_id, payload)


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


@click.group(name="litellm-chatgpt")
def cli() -> None:
    """Manage ChatGPT OAuth profiles for LiteLLM."""


@cli.group(name="profile")
def profile_group() -> None:
    """Manage chatgpt_auth_profiles entries in a LiteLLM config."""


@profile_group.command(name="add")
@click.argument("profile")
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
@click.option(
    "--token-dir",
    help="Token directory for this profile. Defaults to ~/.config/litellm/chatgpt/<profile>.",
)
@click.option(
    "--auth-file",
    help="Optional auth file path or file name for this profile.",
)
@click.option(
    "--with-deployment",
    is_flag=True,
    help="Also add or update a ChatGPT deployment in model_list for this profile.",
)
@click.option(
    "--model-name",
    default="gpt-5.4",
    show_default=True,
    help="Logical model_name for the deployment created by --with-deployment.",
)
@click.option(
    "--provider-model",
    default="chatgpt/gpt-5.4",
    show_default=True,
    help="Provider model for the deployment created by --with-deployment.",
)
@click.option(
    "--mode",
    default="responses",
    show_default=True,
    help="model_info.mode for the deployment created by --with-deployment.",
)
@click.option(
    "--deployment-id",
    help="Optional model_info.id for the deployment. Defaults to chatgpt-<profile>.",
)
def profile_add(
    profile: str,
    config_path: str,
    token_dir: Optional[str],
    auth_file: Optional[str],
    with_deployment: bool,
    model_name: str,
    provider_model: str,
    mode: str,
    deployment_id: Optional[str],
) -> None:
    """Add or update a named ChatGPT auth profile in config."""
    resolved_config = Path(config_path).expanduser().resolve()
    resolved_deployment_id = deployment_id or f"chatgpt-{profile}"

    def _updater(profiles: Dict[str, Any], data: Dict[str, Any]) -> None:
        profile_entry: Dict[str, Any] = {}
        resolved_token_dir = token_dir or _default_token_dir_for_profile(profile)
        profile_entry["token_dir"] = str(Path(resolved_token_dir).expanduser().resolve())
        if auth_file:
            auth_path = Path(auth_file).expanduser()
            profile_entry["auth_file"] = (
                str(auth_path.resolve()) if auth_path.is_absolute() else auth_file
            )
        profiles[profile] = profile_entry

        if with_deployment:
            model_list = _ensure_model_list(data)
            target_deployment: Optional[Dict[str, Any]] = None
            for deployment in model_list:
                if _deployment_matches_profile(deployment, profile):
                    target_deployment = deployment
                    break
            if target_deployment is None:
                target_deployment = {}
                model_list.append(target_deployment)

            existing_model_info = target_deployment.get("model_info")
            if not isinstance(existing_model_info, dict):
                existing_model_info = {}
            existing_litellm_params = target_deployment.get("litellm_params")
            if not isinstance(existing_litellm_params, dict):
                existing_litellm_params = {}

            target_deployment["model_name"] = model_name
            target_deployment["model_info"] = {
                **existing_model_info,
                "id": resolved_deployment_id,
                "mode": mode,
            }
            target_deployment["litellm_params"] = {
                **existing_litellm_params,
                "model": provider_model,
                "chatgpt_auth_profile": profile,
            }

    data, format_name = _load_config_with_format(resolved_config)
    profile_map = data.get("chatgpt_auth_profiles")
    if profile_map is None:
        profile_map = {}
        data["chatgpt_auth_profiles"] = profile_map
    if not isinstance(profile_map, dict):
        raise click.ClickException(
            "chatgpt_auth_profiles must be a mapping in the config file"
        )

    _updater(profile_map, data)
    _save_config_file(resolved_config, data, format_name)
    profiles = profile_map
    added_profile = profiles[profile]
    click.echo(f"updated profile: {profile}")
    click.echo(f"config:          {resolved_config}")
    click.echo(f"token_dir:       {added_profile.get('token_dir')}")
    click.echo(f"auth_file:       {added_profile.get('auth_file', 'auth.json (default)')}")
    if with_deployment:
        click.echo(f"deployment_id:   {resolved_deployment_id}")
        click.echo(f"model_name:      {model_name}")
        click.echo(f"provider_model:  {provider_model}")
    else:
        click.echo(
            "note: this only updates chatgpt_auth_profiles; add a model_list deployment if you want this profile to receive traffic."
        )


@profile_group.command(name="ls")
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit profiles as JSON instead of a table.",
)
def profile_ls(config_path: str, json_output: bool) -> None:
    """List configured ChatGPT auth profiles and linked deployments."""
    resolved_config = Path(config_path).expanduser().resolve()
    data = _load_config_file(resolved_config)
    profiles = data.get("chatgpt_auth_profiles") or {}
    if not isinstance(profiles, dict):
        raise click.ClickException(
            "chatgpt_auth_profiles must be a mapping in the config file"
        )

    rows = []
    for profile in sorted(profiles.keys()):
        profile_entry = profiles[profile] if isinstance(profiles[profile], dict) else {}
        deployments = _get_profile_deployments(data, profile)
        rows.append(
            {
                "profile": profile,
                "token_dir": profile_entry.get("token_dir", "-"),
                "auth_file": profile_entry.get("auth_file", "auth.json (default)"),
                "deployments": [
                    {
                        "model_name": deployment.get("model_name"),
                        "id": (
                            deployment.get("model_info", {}).get("id")
                            if isinstance(deployment.get("model_info"), dict)
                            else None
                        ),
                    }
                    for deployment in deployments
                ],
            }
        )

    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    table_rows = [
        [
            item["profile"],
            str(item["token_dir"]),
            str(item["auth_file"]),
            _format_profile_deployments(_get_profile_deployments(data, item["profile"])),
        ]
        for item in rows
    ]
    click.echo(
        _render_table(
            ["profile", "token_dir", "auth_file", "deployments"],
            table_rows,
        )
    )


@profile_group.command(name="rm")
@click.argument("profile")
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
def profile_rm(profile: str, config_path: str) -> None:
    """Remove a named ChatGPT auth profile from config."""
    resolved_config = Path(config_path).expanduser().resolve()

    def _updater(profiles: Dict[str, Any]) -> None:
        if profile not in profiles:
            raise click.ClickException(
                f"Profile '{profile}' not found in chatgpt_auth_profiles"
            )
        del profiles[profile]

    _update_profiles_in_config(resolved_config, _updater)
    click.echo(f"removed profile: {profile}")
    click.echo(f"config:          {resolved_config}")
    click.echo(
        "note: this does not delete auth files or remove model_list deployments; clean those up separately if needed."
    )


@cli.command(name="login")
@click.argument("profile")
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Backup and remove the current auth.json before logging in again.",
)
def login(profile: str, config_path: str, force: bool) -> None:
    """Login or relogin a ChatGPT OAuth profile for LiteLLM."""
    resolved_config = Path(config_path).expanduser().resolve()
    _prepare_profiles(resolved_config)

    authenticator = get_chatgpt_authenticator({"chatgpt_auth_profile": profile})
    auth_path = Path(authenticator.auth_file)

    click.echo(f"profile: {profile}")
    click.echo(f"config:  {resolved_config}")
    click.echo(f"auth:    {auth_path}")

    if force and auth_path.exists():
        backup_path = auth_path.with_name(
            f"auth.backup-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        shutil.copy2(auth_path, backup_path)
        auth_path.unlink()
        click.echo(f"backed up existing auth to: {backup_path}")

    token = authenticator.get_access_token()
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
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit usage data as JSON instead of a table.",
)
def usage(profile: Optional[str], config_path: str, json_output: bool) -> None:
    """Query usage for one or all configured ChatGPT profiles."""
    resolved_config = Path(config_path).expanduser().resolve()
    profiles = _prepare_profiles(resolved_config)
    profile_names = [profile] if profile else sorted(profiles.keys())
    usage_url = str(
        litellm.get_secret("LITELLM_CHATGPT_USAGE_URL")
        or litellm.get_secret("CHATGPT_USAGE_URL")
        or DEFAULT_USAGE_URL
    )

    results = [_fetch_usage_for_profile(name, usage_url) for name in profile_names]
    if json_output:
        click.echo(json.dumps([asdict(item) for item in results], indent=2))
        return

    click.echo(_render_usage_report(results))


if __name__ == "__main__":
    cli()
