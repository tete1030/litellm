## Why

LiteLLM's built-in routing strategies cover common cases, but they do not fully express account-aware policies such as quota exhaustion avoidance, reset-aware selection, or custom provider rotation logic. We need an explicit extension model so routing policy can evolve without repeatedly forking core router behavior.

## What Changes

- Add a first-class extension point for custom rotation strategies that choose deployments from the healthy pool.
- Define the runtime state and deployment metadata available to those strategies, including quota, cooldown, and provider/account health information.
- Support operator-configured strategy selection and parameters in proxy/router config.
- Specify how custom strategies interact with existing retries, fallback chains, cooldowns, and pre-call checks.

## Capabilities

### New Capabilities
- `custom-rotation-strategies`: Register and run custom deployment-selection strategies for a model group using deployment metadata and live router state.
- `deployment-availability-state`: Expose normalized deployment availability signals needed by custom rotation rules, including quota-related availability and next-eligible timing.

### Modified Capabilities

## Impact

- Affects `litellm/router.py` routing-selection flow and related router configuration surfaces.
- Likely introduces shared state requirements for multi-instance deployments, especially when Redis-backed routing is enabled.
- Requires compatibility rules with built-in routing strategies, fallbacks, retries, and cooldown handling.
- Requires new observability and test coverage for deterministic strategy behavior.
