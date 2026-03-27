## ADDED Requirements

### Requirement: Affinity bindings have bounded lifetime
The system SHALL persist affinity bindings with an explicit expiration policy.

#### Scenario: Binding exceeds TTL
- **WHEN** an affinity binding has passed its configured lifetime
- **THEN** the router MUST treat the next matching request as unbound and create a new binding only after selecting an eligible deployment

### Requirement: Affinity state supports shared deployments
The system SHALL store affinity bindings in shared storage when Redis-backed multi-instance routing is enabled.

#### Scenario: Second instance receives a sticky request
- **WHEN** one LiteLLM instance creates an affinity binding and another LiteLLM instance later handles a request with the same affinity key while Redis-backed routing is enabled
- **THEN** the second instance MUST be able to resolve the existing binding from shared state

### Requirement: Binding updates are atomic
The system SHALL update affinity bindings atomically so concurrent requests do not leave the key mapped to an invalid or partial value.

#### Scenario: Concurrent requests trigger rebinding
- **WHEN** multiple requests for the same affinity key simultaneously determine that the previous deployment is ineligible
- **THEN** the system MUST complete binding updates without leaving corrupted affinity state

### Requirement: Affinity state is namespaced for routing safety
The system SHALL scope affinity bindings so unrelated model groups or routing domains do not accidentally share the same binding.

#### Scenario: Same session key appears on different model groups
- **WHEN** two different model groups receive the same raw session marker
- **THEN** the system MUST keep their affinity bindings logically separate unless configuration explicitly shares the namespace
