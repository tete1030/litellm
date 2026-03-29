# ChatGPT Subscription

Use ChatGPT Pro/Max subscription models through LiteLLM with browser-based OAuth authentication.

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

ChatGPT subscription access supports browser OAuth login by default:

1. LiteLLM prints an OpenAI authorize URL
2. Open it in your browser and approve access
3. Paste the redirected callback URL back into the CLI
4. Tokens are stored locally for reuse

Device-code login is still available as an opt-in fallback.

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

### Login a Named Profile

If you install LiteLLM as a package, you can manage named ChatGPT profiles with the bundled CLI.

Login or relogin a profile:

```bash showLineNumbers
litellm-chatgpt login account-b --config ~/.config/litellm/config.yaml
```

Open the authorize URL automatically in your default browser:

```bash showLineNumbers
litellm-chatgpt login account-b --config ~/.config/litellm/config.yaml --open-browser
```

Add or update a profile entry in your config:

```bash showLineNumbers
litellm-chatgpt profile add account-b --config ~/.config/litellm/config.yaml
```

Add a profile and immediately create a deployment in `model_list` for it:

```bash showLineNumbers
litellm-chatgpt profile add account-b \
  --config ~/.config/litellm/config.yaml \
  --with-deployment \
  --model-name gpt-5.4 \
  --provider-model chatgpt/gpt-5.4
```

List configured profiles and their linked deployments:

```bash showLineNumbers
litellm-chatgpt profile ls --config ~/.config/litellm/config.yaml
```

Remove a profile entry from your config:

```bash showLineNumbers
litellm-chatgpt profile rm account-b --config ~/.config/litellm/config.yaml
```

By default this also removes any pending `browser-login-session.json`, but keeps `auth.json` on disk. To delete both the auth file and the profile directory too:

```bash showLineNumbers
litellm-chatgpt profile rm account-b --config ~/.config/litellm/config.yaml --purge-files
```

Opt into device-code login and back up the existing `auth.json` first:

```bash showLineNumbers
litellm-chatgpt login account-b --config ~/.config/litellm/config.yaml --device --force
```

Query usage for all configured profiles:

```bash showLineNumbers
litellm-chatgpt usage --config ~/.config/litellm/config.yaml
```

Query usage for one profile only:

```bash showLineNumbers
litellm-chatgpt usage account-b --config ~/.config/litellm/config.yaml
```
