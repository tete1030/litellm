# ChatGPT Subscription

Use ChatGPT Pro/Max subscription models through LiteLLM with OAuth device flow authentication.

| Property | Details |
|-------|-------|
| Description | ChatGPT subscription access (Codex + GPT-5.3/5.4 family) via ChatGPT backend API |
| Provider Route on LiteLLM | `chatgpt/` |
| Supported Endpoints | `/responses`, `/chat/completions` (bridged to Responses for supported models) |
| API Reference | https://chatgpt.com |

ChatGPT subscription access is native to the Responses API. Chat Completions requests are bridged to Responses for supported models (for example `chatgpt/gpt-5.4`).

Notes:
- The ChatGPT subscription backend rejects token limit fields (`max_tokens`, `max_output_tokens`, `max_completion_tokens`) and `metadata`. LiteLLM strips these fields for this provider.
- `/v1/chat/completions` honors `stream`. When `stream` is false (default), LiteLLM aggregates the Responses stream into a single JSON response.

## Authentication

ChatGPT subscription access uses an OAuth device code flow:

1. LiteLLM prints a device code and verification URL
2. Open the URL, sign in, and enter the code
3. Tokens are stored locally for reuse

## Usage - LiteLLM Python SDK

### Responses (recommended for Codex models)

```python showLineNumbers title="ChatGPT Responses"
import litellm

response = litellm.responses(
    model="chatgpt/gpt-5.3-codex",
    input="Write a Python hello world"
)

print(response)
```

### Chat Completions (bridged to Responses)

```python showLineNumbers title="ChatGPT Chat Completions"
import litellm

response = litellm.completion(
    model="chatgpt/gpt-5.4",
    messages=[{"role": "user", "content": "Write a Python hello world"}]
)

print(response)
```

## Usage - LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
chatgpt_auth_profiles:
  default:
    token_dir: /Users/example/.config/litellm/chatgpt/default
  account-b:
    token_dir: /Users/example/.config/litellm/chatgpt/account-b

model_list:
  - model_name: chatgpt/gpt-5.4
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.4
      chatgpt_auth_profile: default
  - model_name: chatgpt/gpt-5.4-pro
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.4-pro
      chatgpt_auth_profile: account-b
  - model_name: chatgpt/gpt-5.3-codex
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-codex
      chatgpt_auth_profile: default
  - model_name: chatgpt/gpt-5.3-codex-spark
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-codex-spark
      chatgpt_auth_profile: account-b
  - model_name: chatgpt/gpt-5.3-instant
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-instant
      chatgpt_auth_profile: default
  - model_name: chatgpt/gpt-5.3-chat-latest
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-chat-latest
      chatgpt_auth_profile: account-b
```

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

## Configuration

### Environment Variables

- `CHATGPT_TOKEN_DIR`: Custom token storage directory
- `CHATGPT_AUTH_FILE`: Auth file name (default: `auth.json`)
- `CHATGPT_AUTH_PROFILES_JSON`: JSON object of named auth profiles (alias: `CHATGPT_AUTH_PROFILES`)
- `CHATGPT_API_BASE`: Override API base (default: `https://chatgpt.com/backend-api/codex`)
- `OPENAI_CHATGPT_API_BASE`: Alias for `CHATGPT_API_BASE`
- `CHATGPT_ORIGINATOR`: Override the `originator` header value
- `CHATGPT_USER_AGENT`: Override the `User-Agent` header value
- `CHATGPT_USER_AGENT_SUFFIX`: Optional suffix appended to the `User-Agent` header

### Named Auth Profiles

Use `litellm_params.chatgpt_auth_profile` to bind a deployment to a named ChatGPT OAuth login.

Notes:

- If `chatgpt_auth_profile` is omitted, LiteLLM uses the implicit `default` profile and preserves the legacy `CHATGPT_TOKEN_DIR` / `CHATGPT_AUTH_FILE` behavior.
- Each profile can define `token_dir` and/or `auth_file`.
- Proxy startup validates that every referenced `chatgpt_auth_profile` exists.
- Auth state is isolated per profile, so multiple ChatGPT accounts can coexist in one proxy.

Example environment registry:

```bash showLineNumbers
export CHATGPT_AUTH_PROFILES_JSON='{
  "default": {"token_dir": "~/.config/litellm/chatgpt/default"},
  "account-b": {"token_dir": "~/.config/litellm/chatgpt/account-b"}
}'
```
