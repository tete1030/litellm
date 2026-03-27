from unittest.mock import MagicMock, patch

import litellm


@patch("litellm.completion_extras.responses_api_bridge.completion")
@patch("litellm.llms.chatgpt.authenticator.get_chatgpt_authenticator")
def test_chatgpt_completion_uses_profile_access_token(
    mock_get_chatgpt_authenticator, mock_completion
):
    mock_authenticator = MagicMock()
    mock_authenticator.get_access_token.return_value = "profile-token"
    mock_authenticator.get_api_base.return_value = "https://chatgpt.example.com"
    mock_authenticator.get_account_id.return_value = "acct-123"
    mock_get_chatgpt_authenticator.return_value = mock_authenticator
    mock_completion.return_value = MagicMock()

    litellm.completion(
        model="chatgpt/gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        chatgpt_auth_profile="account-a",
    )

    assert mock_completion.call_args.kwargs["api_key"] == "profile-token"
