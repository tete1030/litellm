## Context

Several LiteLLM use cases benefit from keeping related requests on the same deployment: provider-native prompt caching, continuation of encrypted Responses API content, and application-level session locality. Existing LiteLLM features provide partial affinity primitives, but there is no unified routing capability that treats session IDs or cache keys as first-class deployment-affinity inputs.

## Goals / Non-Goals

**Goals:**
- Define an explicit affinity key model based on stable request markers such as session IDs and cache keys.
- Keep requests on the same deployment while that deployment remains eligible.
- Support rebinding when the bound deployment is unavailable, cooled down, or no longer policy-eligible.
- Work in both local and Redis-backed multi-instance deployments.

**Non-Goals:**
- Infer affinity purely from prompt text without an explicit marker.
- Guarantee affinity across deployments that cannot decrypt or reuse one another's provider-specific artifacts.
- Replace existing fallback or retry mechanisms.

## Decisions

### Normalize request markers into one affinity key
The router will derive a single affinity key from configured request markers such as `metadata.session_id`, `prompt_cache_key`, or other explicitly supported fields. A unified key lets storage and routing semantics stay simple while still supporting multiple upstream marker types.

Alternative considered: maintain separate affinity systems per marker type. Rejected because it creates conflicting binding logic and duplicate state.

### Evaluate affinity before general rotation, but after hard eligibility checks
If a request resolves to an existing affinity binding and the bound deployment is still eligible, the router should prefer it before invoking general rotation strategy selection. Hard filters such as explicit disablement, cooldown, or incompatible encrypted-content constraints remain authoritative.

Alternative considered: run general rotation first and use affinity only as a tiebreaker. Rejected because it weakens cache locality and session stickiness.

### Rebind explicitly when the bound deployment is no longer eligible
When the bound deployment cannot serve the request, the router should select a new eligible deployment using existing routing logic, then atomically update the affinity binding. This keeps failover intact while preserving stickiness for later calls.

Alternative considered: fail all bound requests until the original deployment recovers. Rejected because it defeats LiteLLM's reliability goals.

### Store bindings in local or Redis-backed affinity state
Single-instance deployments can use process-local binding state, while Redis-backed deployments will persist affinity bindings in shared storage with TTL. This aligns affinity semantics with current multi-instance router patterns.

Alternative considered: require Redis for all affinity use. Rejected because it would block smaller deployments from using the feature.

## Risks / Trade-offs

- [Affinity can reduce even load distribution] -> Let operators configure TTLs, allowed markers, and rebinding behavior.
- [Bad client markers can create hot spots] -> Make affinity opt-in and document expected marker cardinality.
- [Shared-state lookups add routing overhead] -> Keep key derivation simple and support local-state mode when Redis is absent.
- [Affinity semantics can conflict with custom rotation] -> Define precedence: eligibility checks first, then affinity, then custom rotation among remaining candidates.

## Migration Plan

1. Add request-marker parsing and affinity key derivation.
2. Introduce binding storage with TTL for local and Redis-backed modes.
3. Integrate affinity lookup and rebinding into router selection flow.
4. Add tests for sticky routing, rebinding, and fallback/cooldown interaction.
5. Rollback path: disable affinity configuration and return to normal load balancing.

## Open Questions

- Which marker sources should be supported in the initial release beyond `metadata.session_id` and `prompt_cache_key`?
- Should rebinding events surface through headers, logs, or metrics in the first version?
- Do we need per-model-group affinity namespaces by default, or should operators choose the namespace boundary?
