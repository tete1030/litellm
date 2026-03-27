## Context

LiteLLM's `chatgpt/` provider currently assumes one OAuth login state per process, driven by global token-path environment variables and a single authenticator flow. That prevents operators from modeling multiple ChatGPT accounts as separate deployments, and it also makes refresh behavior fragile when several requests or workers touch the same auth state.

## Goals / Non-Goals

**Goals:**
- Let multiple ChatGPT OAuth accounts coexist in one LiteLLM deployment.
- Bind each deployment to a specific auth profile without changing router semantics for non-ChatGPT providers.
- Preserve backward compatibility for current single-account users.
- Make per-profile token reads, refreshes, and writes concurrency-safe.

**Non-Goals:**
- Generalize multi-account auth for every provider in the same change.
- Introduce a new external secret manager requirement.
- Redesign LiteLLM's router or load-balancing algorithms beyond what is needed to expose multiple ChatGPT-backed deployments.

## Decisions

### Use deployment-level auth profiles
Each `chatgpt/*` deployment will reference a named auth profile such as `chatgpt_auth_profile: primary`. The profile abstraction is more stable than raw path overrides, lets config stay readable, and keeps room for future profile-level validation.

Alternative considered: pass only file paths on each deployment. Rejected because it duplicates storage configuration, makes validation harder, and couples deployment config directly to filesystem layout.

### Resolve profiles through a profile registry with a default profile
The provider will build a profile registry that maps profile names to token directories or auth files. If a deployment omits `chatgpt_auth_profile`, LiteLLM will resolve the existing single-account location as the `default` profile so older configs keep working.

Alternative considered: require every deployment to name a profile. Rejected because it would create an unnecessary breaking migration.

### Cache authenticators per profile, not per provider
Request paths will obtain an authenticator from a registry keyed by resolved profile identity. This keeps token refresh state isolated per account while still reusing authenticator instances for performance.

Alternative considered: instantiate a fresh authenticator on every request. Rejected because it increases overhead and makes lock/state coordination harder.

### Make refresh persistence safe per profile
Each profile will use atomic writes and a per-profile lock around refresh/update operations. Local single-process safety is required in the initial implementation, and the design keeps room for a file-lock or distributed-lock extension if multi-process deployments need stronger guarantees.

Alternative considered: rely on existing best-effort file writes. Rejected because multi-account support increases concurrent use of the same account profile and makes latent refresh races more visible.

## Risks / Trade-offs

- [Profile config grows more complex] -> Keep a default profile path and document the minimum config needed for two-account setups.
- [Atomic writes and locking add implementation complexity] -> Limit the first iteration to per-profile semantics and cover refresh races with tests.
- [Operators may expect multi-process locking on day one] -> Document the guarantee boundary and keep lock abstraction extensible.
- [OAuth tokens remain file-backed state] -> Keep storage format compatible so operators can still inspect and back up credentials.

## Migration Plan

1. Introduce profile-aware config fields and default-profile resolution.
2. Refactor ChatGPT request paths to resolve authenticators through the profile registry.
3. Add safe per-profile persistence and refresh tests.
4. Document migration from single-account layout to named profiles.
5. Rollback path: remove profile config and fall back to implicit default profile behavior.

## Open Questions

- Should profile definitions live in a dedicated top-level config block or stay entirely deployment-local in the first version?
- Do we need cross-process file locks in the initial formal release, or is atomic replace plus documented single-host expectations sufficient?
- Should the same profile abstraction be introduced for `github_copilot/` in a follow-up change?
