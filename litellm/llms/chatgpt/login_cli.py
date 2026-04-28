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

import click

import litellm

from .authenticator import BrowserLoginSession, get_chatgpt_authenticator
from . import usage_service

ChatGPTUsageMetrics = usage_service.ChatGPTUsageMetrics
UsageResult = usage_service.UsageResult
UsageWindow = usage_service.UsageWindow
_fetch_usage_for_profile = usage_service.fetch_usage_for_profile
_get_usage_url = usage_service.get_usage_url
_normalize_usage_payload = usage_service.normalize_usage_payload

DEFAULT_CONFIG_ENV_VAR = "CONFIG_FILE_PATH"
DEFAULT_CONFIG_DIR = Path.home() / ".config/litellm"
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG_CANDIDATES = (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_DIR / "config.yml",
)
DEFAULT_CONFIG_HELP = (
    "$CONFIG_FILE_PATH, ~/.config/litellm/config.yaml, ~/.config/litellm/config.yml"
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
        raise click.ClickException(f"Config not found: {resolved}")
    return resolved


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
    suffix = config_path.suffix.lower()
    format_name = "json" if suffix == ".json" else "yaml"
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
                "Editing YAML config files requires ruamel.yaml to preserve comments. "
                "Install with `pip install 'litellm[proxy]'` or use a JSON config file."
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


def _refresh_usage_metrics(
    config_path: Path, usage_url: str, metrics: ChatGPTUsageMetrics
) -> List[UsageResult]:
    profiles = _prepare_profiles(config_path)
    results = [_fetch_usage_for_profile(name, usage_url) for name in sorted(profiles.keys())]
    metrics.update(results)
    return results


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
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
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
    resolved_config = _resolve_config_path(config_path)
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
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
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
    resolved_config = _resolve_config_path(config_path)
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
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
)
@click.option(
    "--purge-files",
    is_flag=True,
    help="Also delete the profile auth.json and remove the profile directory if it becomes empty.",
)
def profile_rm(profile: str, config_path: str, purge_files: bool) -> None:
    """Remove a named ChatGPT auth profile from config."""
    resolved_config = _resolve_config_path(config_path)
    profiles = _prepare_profiles(resolved_config)
    if profile not in profiles:
        raise click.ClickException(
            f"Profile '{profile}' not found in chatgpt_auth_profiles"
        )

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
    click.echo(f"config:          {resolved_config}")
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
    click.echo(
        "note: model_list deployments are not removed automatically; clean those up separately if needed."
    )


@cli.command(name="login")
@click.argument("profile")
@click.option(
    "--config",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
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
    click.echo(f"config:  {resolved_config}")
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
    "--config",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
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
    resolved_config = _resolve_config_path(config_path)
    profiles = _prepare_profiles(resolved_config)
    profile_names = [profile] if profile else sorted(profiles.keys())
    usage_url = _get_usage_url()

    results = [_fetch_usage_for_profile(name, usage_url) for name in profile_names]
    if json_output:
        click.echo(json.dumps([asdict(item) for item in results], indent=2))
        return

    click.echo(_render_usage_report(results))


@cli.command(name="metrics")
@click.option(
    "--config",
    "config_path",
    default=None,
    show_default=DEFAULT_CONFIG_HELP,
    help="LiteLLM config file containing chatgpt_auth_profiles.",
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
    help="Refresh interval for ChatGPT usage polling.",
)
def metrics(config_path: Optional[str], listen_host: str, port: int, interval_seconds: int) -> None:
    """Serve Prometheus metrics for ChatGPT account usage windows."""
    if interval_seconds <= 0:
        raise click.ClickException("--interval-seconds must be greater than 0")

    resolved_config = _resolve_config_path(config_path)
    usage_url = _get_usage_url()

    try:
        from prometheus_client import CollectorRegistry, start_http_server
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "Prometheus metrics require prometheus_client. Install with `pip install prometheus-client`."
        ) from exc

    registry = CollectorRegistry()
    usage_metrics = ChatGPTUsageMetrics(registry=registry)

    results = _refresh_usage_metrics(resolved_config, usage_url, usage_metrics)
    start_http_server(port, addr=listen_host, registry=registry)
    click.echo(
        f"serving ChatGPT usage metrics on http://{listen_host}:{port}/metrics "
        f"for {len(results)} profile(s); refresh interval {interval_seconds}s"
    )

    while True:
        try:
            time.sleep(interval_seconds)
            _refresh_usage_metrics(resolved_config, usage_url, usage_metrics)
        except KeyboardInterrupt:
            click.echo("stopping ChatGPT usage metrics exporter")
            return
        except Exception as exc:
            usage_metrics.mark_refresh_failure()
            click.echo(f"warning: failed refreshing ChatGPT usage metrics: {exc}", err=True)


if __name__ == "__main__":
    cli()
