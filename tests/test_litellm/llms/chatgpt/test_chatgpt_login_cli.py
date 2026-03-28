from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from litellm.llms.chatgpt.login_cli import UsageResult, UsageWindow, cli


def test_chatgpt_login_cli_reads_yaml_config_and_logs_in(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chatgpt_auth_profiles:\n  buy2:\n    token_dir: /tmp/buy2\n"
    )

    mock_authenticator = MagicMock()
    mock_authenticator.auth_file = str(tmp_path / "buy2" / "auth.json")
    mock_authenticator.get_access_token.return_value = "token-123"
    mock_authenticator.get_account_id.return_value = "acct-123"

    auth_path = Path(mock_authenticator.auth_file)
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"expires_at": 1234567890}))

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ) as mock_get_authenticator:
        result = runner.invoke(cli, ["login", "buy2", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "login complete" in result.output
    assert "account_id: acct-123" in result.output
    mock_get_authenticator.assert_called_once_with({"chatgpt_auth_profile": "buy2"})


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
    mock_authenticator.get_access_token.return_value = "token-123"
    mock_authenticator.get_account_id.return_value = "acct-777"

    runner = CliRunner()
    with patch(
        "litellm.llms.chatgpt.login_cli.get_chatgpt_authenticator",
        return_value=mock_authenticator,
    ):
        result = runner.invoke(
            cli, ["login", "buy7", "--config", str(config_path), "--force"]
        )

    assert result.exit_code == 0
    assert "backed up existing auth to:" in result.output
    backups = list(auth_path.parent.glob("auth.backup-*.json"))
    assert len(backups) == 1


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
