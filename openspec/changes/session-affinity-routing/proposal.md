## Why

Load balancing alone can reduce cache hit rates and break request continuity when follow-up calls should stay on the same deployment for prompt-cache locality, encrypted response continuity, or application-level session stickiness. We need a defined affinity model so operators can preserve locality without disabling failover.

## What Changes

- Add affinity-aware routing based on stable request markers such as session IDs and cache keys.
- Define how affinity bindings are created, stored, expired, and rebound when a deployment becomes unavailable.
- Specify how affinity cooperates with fallback, cooldown, retries, and custom rotation strategies.
- Document the request metadata contract clients or upstream proxies should use to participate in affinity routing.

## Capabilities

### New Capabilities
- `session-affinity-routing`: Keep requests with the same affinity marker on the same deployment while it remains eligible, with controlled rebinding when it does not.
- `affinity-state-management`: Persist and expire affinity bindings consistently across single-instance and Redis-backed multi-instance deployments.

### Modified Capabilities

## Impact

- Affects router selection flow, request metadata handling, and shared-state storage.
- Interacts with Responses API affinity, prompt-cache hints, and any future cache-aware routing features.
- Requires explicit conflict-resolution rules with cooldowns, rate-limit enforcement, retries, and fallbacks.
- Requires new docs and tests for session markers, TTL behavior, and rebinding semantics.
