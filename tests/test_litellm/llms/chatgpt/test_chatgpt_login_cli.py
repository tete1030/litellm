from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from prometheus_client import CollectorRegistry, generate_latest

from litellm.llms.chatgpt.authenticator import BrowserLoginSession
from litellm.llms.chatgpt.login_cli import (
    ChatGPTUsageMetrics,
    UsageResult,
    UsageWindow,
    _normalize_usage_payload,
    _resolve_config_path,
    cli,
)


def test_chatgpt_login_cli_reads_yaml_config_and_logs_in(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    mock_authenticator = MagicMock()
    mock_authenticator.auth_file = str(tmp_path / "buy2" / "auth.json")
    mock_authenticator.create_browser_login_session.return_value = BrowserLoginSession(
        authorize_url="https://auth.openai.com/oauth/authorize?foo=bar",
        redirect_uri="http://localhost:1455/auth/callback",
        state="xyz",
        code_verifier="verifier-123",
    )

    auth_path = Path(mock_authenticator.auth_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"expires_at": 1234567890}))

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ) as mock_get_authenticator:
        start_result = runner.invoke(
            cli,
            [
                "login",
                "buy2",
                "--config",
                str(config_path),
            ],
        )

        session_path = auth_path.with_name("browser-login-session.json")
        assert session_path.exists()

        mock_authenticator.complete_browser_login.return_value = {
            "access_token": "token-123"
        }
        mock_authenticator.get_account_id.return_value = "acct-123"

        complete_result = runner.invoke(
            cli,
            [
                "login",
                "buy2",
                "--config",
                str(config_path),
                "--callback-url",
                "http://localhost:1455/auth/callback?code=abc&state=xyz",
            ],
        )

    assert start_result.exit_code == 0
    assert "browser login url:" in start_result.output
    assert "session:" in start_result.output
    assert complete_result.exit_code == 0
    assert "browser login complete" in complete_result.output
    assert "login complete" in complete_result.output
    assert "account_id: acct-123" in complete_result.output
    assert not session_path.exists()
    assert mock_get_authenticator.call_count == 2
    assert mock_get_authenticator.call_args_list[0].args == (
        {"chatgpt_auth_profile": "buy2"},
    )
    assert mock_get_authenticator.call_args_list[1].args == (
        {"chatgpt_auth_profile": "buy2"},
    )
    mock_authenticator.create_browser_login_session.assert_called_once_with(
        redirect_uri=None, allowed_workspace_id=None
    )
    login_session = mock_authenticator.complete_browser_login.call_args.args[0]
    assert login_session.state == "xyz"
    mock_authenticator.complete_browser_login.assert_called_once()


def test_chatgpt_login_cli_force_backs_up_existing_auth(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy7:\n    token_dir: /tmp/buy7\n"
    )

    auth_path = tmp_path / "buy7" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"expires_at": 1}))

    mock_authenticator = MagicMock()
    mock_authenticator.auth_file = str(auth_path)
    mock_authenticator.create_browser_login_session.return_value = BrowserLoginSession(
        authorize_url="https://auth.openai.com/oauth/authorize?foo=bar",
        redirect_uri="http://localhost:1455/auth/callback",
        state="xyz",
        code_verifier="verifier-123",
    )

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ):
        result = runner.invoke(
            cli,
            [
                "login",
                "buy7",
                "--config",
                str(config_path),
                "--force",
            ],
        )

    assert result.exit_code == 0
    assert "backed up existing auth to:" in result.output
    backups = list(auth_path.parent.glob("auth.backup-*.json"))
    assert len(backups) == 1


def test_chatgpt_login_cli_device_mode_is_opt_in(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy3:\n    token_dir: /tmp/buy3\n"
    )

    mock_authenticator = MagicMock()
    mock_authenticator.auth_file = str(tmp_path / "buy3" / "auth.json")
    mock_authenticator._login_device_code.return_value = {"access_token": "token-123"}
    mock_authenticator.get_account_id.return_value = "acct-333"

    auth_path = Path(mock_authenticator.auth_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"expires_at": 1234567890}))

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ):
        result = runner.invoke(
            cli, ["login", "buy3", "--config", str(config_path), "--device"]
        )

    assert result.exit_code == 0
    assert "using device-code login" in result.output
    mock_authenticator._login_device_code.assert_called_once_with()
    mock_authenticator.create_browser_login_session.assert_not_called()


def test_chatgpt_login_cli_complete_requires_pending_session(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    mock_authenticator = MagicMock()
    mock_authenticator.auth_file = str(tmp_path / "buy2" / "auth.json")

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ):
        result = runner.invoke(
            cli,
            [
                "login",
                "buy2",
                "--config",
                str(config_path),
                "--callback-url",
                "http://localhost:1455/auth/callback?code=abc&state=xyz",
            ],
        )

    assert result.exit_code != 0
    assert "No pending browser login session found" in result.output


def test_chatgpt_login_cli_requires_config_file(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"
    runner = CliRunner()

    result = runner.invoke(cli, ["login", "buy1", "--config", str(missing_config)])

    assert result.exit_code != 0
    assert f"Config not found: {missing_config}" in result.output


def test_chatgpt_usage_cli_prints_all_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n  my:\n    token_dir: /tmp/my\n"
    )

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli._fetch_usage_for_profile",
        side_effect=[
            UsageResult(
                profile="buy2",
                account_id="acct-buy2-1234",
                plan="plus",
                credits_balance=0.0,
                windows=[UsageWindow(label="3h", used_percent=5.0, reset_at=None)],
                status="ok",
            ),
            UsageResult(
                profile="my",
                account_id="acct-my-5678",
                plan="team",
                credits_balance=12.5,
                windows=[],
                status="ok",
            ),
        ],
    ) as mock_fetch_usage:
        result = runner.invoke(cli, ["usage", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "profile" in result.output
    assert "buy2" in result.output
    assert "my" in result.output
    assert "plus" in result.output
    assert "team" in result.output
    assert mock_fetch_usage.call_count == 2


def test_chatgpt_usage_cli_can_emit_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli._fetch_usage_for_profile",
        return_value=UsageResult(
            profile="buy2",
            account_id="acct-buy2-1234",
            plan="plus",
            credits_balance=0.0,
            windows=[UsageWindow(label="3h", used_percent=5.0, reset_at=1700000000)],
            status="ok",
        ),
    ):
        result = runner.invoke(cli, ["usage", "buy2", "--config", str(config_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["profile"] == "buy2"
    assert payload[0]["plan"] == "plus"


def test_chatgpt_usage_cli_uses_config_file_path_env_var(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli._fetch_usage_for_profile",
        return_value=UsageResult(
            profile="buy2",
            account_id="acct-buy2-1234",
            plan="plus",
            credits_balance=0.0,
            windows=[UsageWindow(label="3h", used_percent=5.0, reset_at=1700000000)],
            status="ok",
        ),
    ):
        result = runner.invoke(
            cli,
            ["usage", "buy2", "--json"],
            env={"CONFIG_FILE_PATH": str(config_path)},
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["profile"] == "buy2"


def test_chatgpt_resolve_config_prefers_yaml_before_yml(tmp_path: Path) -> None:
    standard_config = tmp_path / "config.yaml"
    secondary_config = tmp_path / "config.yml"
    standard_config.write_text("chatgpt_auth_profiles: {}\n")
    secondary_config.write_text("chatgpt_auth_profiles: {}\n")

    with patch("litellm.llms.chatgpt.login_cli.DEFAULT_CONFIG", standard_config), patch(
        "litellm.llms.chatgpt.login_cli.DEFAULT_CONFIG_CANDIDATES",
        (standard_config, secondary_config),
    ):
        resolved = _resolve_config_path(None)

    assert resolved == standard_config.resolve()


def test_chatgpt_usage_payload_labels_weekly_windows() -> None:
    result = _normalize_usage_payload(
        profile="buy2",
        account_id="acct-buy2-1234",
        payload={
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "limit_window_seconds": 18000,
                    "used_percent": 25,
                    "reset_at": 1700000000,
                },
                "secondary_window": {
                    "limit_window_seconds": 604800,
                    "used_percent": 80,
                    "reset_at": 1700500000,
                },
            },
        },
    )

    assert [window.label for window in result.windows] == ["5h", "1w"]
    assert [window.limit_seconds for window in result.windows] == [18000, 604800]


def test_chatgpt_usage_metrics_exporter_updates_prometheus_gauges() -> None:
    registry = CollectorRegistry()
    metrics = ChatGPTUsageMetrics(registry=registry)
    metrics.update(
        [
            UsageResult(
                profile="buy2",
                account_id="acct-buy2-1234",
                plan="plus",
                account_type="plus",
                has_active_subscription=True,
                subscription_expires_at=1779267024,
                subscription_renews_at=1779177024,
                effective_available=True,
                credits_balance=12.5,
                windows=[
                    UsageWindow(
                        label="5h",
                        used_percent=40.0,
                        reset_at=1700003600,
                        limit_seconds=18000,
                    ),
                    UsageWindow(
                        label="1w",
                        used_percent=80.0,
                        reset_at=1700600000,
                        limit_seconds=604800,
                    ),
                ],
                status="ok",
            )
        ],
        refreshed_at=1700000000,
    )

    payload = generate_latest(registry).decode("utf-8")
    assert 'litellm_chatgpt_profile_up{profile="buy2"} 1.0' in payload
    assert 'litellm_chatgpt_profile_available{profile="buy2"} 1.0' in payload
    assert 'litellm_chatgpt_profile_has_active_subscription{profile="buy2"} 1.0' in payload
    assert 'litellm_chatgpt_profile_subscription_expires_timestamp_seconds{profile="buy2"} 1.779267024e+09' in payload
    assert 'litellm_chatgpt_profile_subscription_renews_timestamp_seconds{profile="buy2"} 1.779177024e+09' in payload
    assert 'litellm_chatgpt_profile_plan_info{account_type="plus",profile="buy2"} 1.0' in payload
    assert (
        'litellm_chatgpt_usage_window_limit_seconds{profile="buy2",window="5h"} 18000.0'
        in payload
    )
    assert (
        'litellm_chatgpt_usage_window_remaining_seconds{profile="buy2",window="1w"} 600000.0'
        in payload
    )


def test_chatgpt_profile_add_updates_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("chatgpt_auth_profiles:\n  buy1:\n    token_dir: /tmp/buy1\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "add", "buy8", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "updated profile: buy8" in result.output
    data = config_path.read_text()
    assert "buy8:" in data
    assert str((Path.home() / ".config/litellm/chatgpt/buy8").resolve()) in data


def test_chatgpt_profile_add_preserves_yaml_comments(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "# top-level comment\n"
        "chatgpt_auth_profiles:\n"
        "  # existing profile comment\n"
        "  buy1: # inline profile comment\n"
        "    token_dir: /tmp/buy1\n"
        "# model list comment\n"
        "model_list: []\n"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "add", "buy8", "--config", str(config_path)])

    assert result.exit_code == 0
    data = config_path.read_text()
    assert "# top-level comment" in data
    assert "# existing profile comment" in data
    assert "# inline profile comment" in data
    assert "# model list comment" in data
    assert "buy8:" in data


def test_chatgpt_profile_add_updates_json_config_with_custom_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"chatgpt_auth_profiles": {}}))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "profile",
            "add",
            "buy9",
            "--config",
            str(config_path),
            "--token-dir",
            str(tmp_path / "tokens" / "buy9"),
            "--auth-file",
            "custom.json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(config_path.read_text())
    assert payload["chatgpt_auth_profiles"]["buy9"]["token_dir"] == str(
        (tmp_path / "tokens" / "buy9").resolve()
    )
    assert payload["chatgpt_auth_profiles"]["buy9"]["auth_file"] == "custom.json"


def test_chatgpt_profile_add_with_deployment_updates_model_list(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("chatgpt_auth_profiles: {}\nmodel_list: []\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "profile",
            "add",
            "buy10",
            "--config",
            str(config_path),
            "--with-deployment",
            "--model-name",
            "gpt-5.4-mini-pool",
            "--provider-model",
            "chatgpt/gpt-5.4-mini",
        ],
    )

    assert result.exit_code == 0
    payload = config_path.read_text()
    assert "buy10:" in payload
    assert "gpt-5.4-mini-pool" in payload
    assert "chatgpt-buy10" in payload
    assert "chatgpt/gpt-5.4-mini" in payload


def test_chatgpt_profile_rm_removes_profile_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy1:\n    token_dir: /tmp/buy1\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "rm", "buy2", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "removed profile: buy2" in result.output
    data = config_path.read_text()
    assert "buy1:" in data
    assert "buy2:" not in data


def test_chatgpt_profile_rm_deletes_pending_browser_session_by_default(
    tmp_path: Path,
) -> None:
    token_dir = tmp_path / "buy2"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"chatgpt_auth_profiles:\n  buy2:\n    token_dir: {token_dir}\n"
    )
    token_dir.mkdir(parents=True, exist_ok=True)
    session_path = token_dir / "browser-login-session.json"
    auth_path = token_dir / "auth.json"
    session_path.write_text(json.dumps({"state": "xyz"}))
    auth_path.write_text(json.dumps({"access_token": "token"}))

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "rm", "buy2", "--config", str(config_path)])

    assert result.exit_code == 0
    assert not session_path.exists()
    assert auth_path.exists()
    assert "removed session:" in result.output
    assert "auth.json is kept by default" in result.output


def test_chatgpt_profile_rm_can_purge_auth_files(tmp_path: Path) -> None:
    token_dir = tmp_path / "buy2"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"chatgpt_auth_profiles:\n  buy2:\n    token_dir: {token_dir}\n"
    )
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / "browser-login-session.json").write_text(json.dumps({"state": "xyz"}))
    (token_dir / "auth.json").write_text(json.dumps({"access_token": "token"}))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["profile", "rm", "buy2", "--config", str(config_path), "--purge-files"],
    )

    assert result.exit_code == 0
    assert not token_dir.exists()
    assert "removed auth:" in result.output
    assert "removed dir:" in result.output


def test_chatgpt_profile_rm_requires_existing_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("chatgpt_auth_profiles:\n  buy1:\n    token_dir: /tmp/buy1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli, ["profile", "rm", "missing", "--config", str(config_path)]
    )

    assert result.exit_code != 0
    assert "Profile 'missing' not found in chatgpt_auth_profiles" in result.output


def test_chatgpt_profile_ls_prints_profiles_and_deployments(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n"
        "  buy1:\n"
        "    token_dir: /tmp/buy1\n"
        "  buy2:\n"
        "    token_dir: /tmp/buy2\n"
        "model_list:\n"
        "  - model_name: gpt-5.4\n"
        "    model_info:\n"
        "      id: chatgpt-buy1\n"
        "      mode: responses\n"
        "    litellm_params:\n"
        "      model: chatgpt/gpt-5.4\n"
        "      chatgpt_auth_profile: buy1\n"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "ls", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "buy1" in result.output
    assert "buy2" in result.output
    assert "chatgpt-buy1 (gpt-5.4)" in result.output


def test_chatgpt_profile_ls_can_emit_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n"
        "  buy1:\n"
        "    token_dir: /tmp/buy1\n"
        "model_list:\n"
        "  - model_name: gpt-5.4\n"
        "    model_info:\n"
        "      id: chatgpt-buy1\n"
        "    litellm_params:\n"
        "      model: chatgpt/gpt-5.4\n"
        "      chatgpt_auth_profile: buy1\n"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "ls", "--config", str(config_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["profile"] == "buy1"
    assert payload[0]["deployments"][0]["id"] == "chatgpt-buy1"
