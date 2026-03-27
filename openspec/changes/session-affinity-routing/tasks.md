## 1. Affinity Markers and State

- [ ] 1.1 Define supported affinity markers and implement stable affinity-key derivation.
- [ ] 1.2 Implement affinity binding storage with TTL, namespacing, and local/Redis-backed backends.

## 2. Router Integration

- [ ] 2.1 Integrate affinity lookup after hard eligibility checks and before general rotation.
- [ ] 2.2 Implement atomic rebinding when a bound deployment is ineligible or unavailable.
- [ ] 2.3 Ensure affinity-routed requests continue to use existing retry, fallback, and cooldown behavior.

## 3. Verification and Documentation

- [ ] 3.1 Add tests for sticky routing, TTL expiry, rebinding, and cross-instance Redis behavior.
- [ ] 3.2 Add tests for interactions with cooldowns, encrypted-content constraints, and custom rotation strategies.
- [ ] 3.3 Document request-marker contracts, operator configuration, and expected rebinding semantics.
