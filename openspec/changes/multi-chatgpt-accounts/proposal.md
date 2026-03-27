## Why

LiteLLM's current `chatgpt/` provider behaves like a single-login integration, which prevents one deployment group from using multiple ChatGPT OAuth accounts for rotation, fallback, and load balancing. We need a formal multi-account model now so ChatGPT-backed capacity can scale the same way other multi-deployment providers already do.

## What Changes

- Add deployment-level ChatGPT auth profiles so multiple OAuth accounts can coexist under the same logical model group.
- Define isolated auth-state storage, refresh, and validation behavior for each profile while preserving backward compatibility for existing single-account configs.
- Support routing and failover across deployments backed by different ChatGPT accounts.
- Document the operational model for profile provisioning, token persistence, and migration from the current single-account layout.

## Capabilities

### New Capabilities
- `chatgpt-account-profiles`: Define and use multiple ChatGPT OAuth account profiles, each with independent credentials and token state, across one or more deployments.

### Modified Capabilities

## Impact

- Affects `litellm/llms/chatgpt/` authentication and request-transformation code paths.
- Affects deployment config schema and validation for `chatgpt/*` models.
- Requires concurrency-safe token persistence semantics for per-profile auth state.
- Requires new tests and docs for configuration, migration, and failure handling.
