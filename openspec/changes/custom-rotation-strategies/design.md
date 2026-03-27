## Context

LiteLLM already provides several built-in routing strategies, cooldowns, fallbacks, retries, and pre-call checks. That is enough for general balancing, but not for policies such as reset-aware account selection, custom quota routing, or application-specific prioritization rules across many equivalent deployments.

## Goals / Non-Goals

**Goals:**
- Expose a formal extension point for custom deployment-selection logic.
- Make normalized deployment availability state available to those strategies.
- Preserve existing router safeguards such as health filtering, cooldowns, and fallback handling.
- Support single-instance and Redis-backed multi-instance operation.

**Non-Goals:**
- Replace existing built-in routing strategies.
- Standardize every provider-specific rate-limit header in the first change.
- Add a user-facing rule DSL in the first version.

## Decisions

### Use a strategy interface that consumes candidate deployments and request context
Custom rotation will be modeled as a routing strategy interface that receives the model group, request kwargs, healthy candidate deployments, and a normalized state snapshot. This keeps the integration close to current router behavior and lets strategies stay pure selection logic.

Alternative considered: add many new top-level config flags for every rotation heuristic. Rejected because it would not scale to provider-specific or operator-specific policies.

### Normalize availability state before strategy execution
The router will compute a normalized availability snapshot per deployment containing signals such as cooldown status, remaining quota if known, retry-after timing if known, and any derived next-eligible timestamp. Strategies can then reason about one stable model rather than raw provider responses.

Alternative considered: let strategies parse provider-specific headers directly. Rejected because it pushes transport details into every strategy and makes reuse difficult.

### Keep router safety checks authoritative
Custom strategies may rank or choose among eligible candidates, but the router remains responsible for authoritative filtering, retries, fallbacks, and cooldown enforcement. This avoids letting strategy code bypass core reliability guarantees.

Alternative considered: give strategies full control of all routing and fallback behavior. Rejected because it would make failure handling inconsistent and harder to debug.

### Use Redis-backed state when multi-instance routing is enabled
When Redis is configured, availability state and any strategy-required shared counters should be stored in Redis so routing behavior is consistent across proxy instances. Without Redis, the router will fall back to process-local state.

Alternative considered: keep all state local even in multi-instance mode. Rejected because reset-aware or quota-aware routing would diverge across replicas.

## Risks / Trade-offs

- [Custom strategies may be nondeterministic or slow] -> Define a small, synchronous/async-safe contract and add timing/error guardrails.
- [Normalized state may not capture every provider nuance] -> Version the state contract and allow additive fields over time.
- [Redis dependence grows for advanced routing] -> Keep local mode supported and document where multi-instance consistency requires Redis.
- [Debugging custom selection can be hard] -> Include observability hooks that record chosen strategy and selected deployment.

## Migration Plan

1. Add the custom strategy interface and config surface.
2. Build normalized deployment-availability state and expose it to strategies.
3. Integrate strategy selection into router candidate picking without changing fallback semantics.
4. Add observability and tests for selection, failure, and multi-instance behavior.
5. Rollback path: switch back to a built-in strategy in config.

## Open Questions

- Should custom strategies return one deployment or an ordered candidate list?
- Which provider-specific reset signals are in scope for the first normalized state contract?
- Should strategy errors hard-fail the request or automatically fall back to a configured built-in strategy?
