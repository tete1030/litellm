## ADDED Requirements

### Requirement: Deployment availability is exposed in a normalized form
The system SHALL expose deployment availability signals to custom rotation strategies through a normalized state model.

#### Scenario: Strategy reads normalized state
- **WHEN** a custom rotation strategy evaluates candidate deployments
- **THEN** each candidate MUST include normalized availability fields for health, cooldown status, and any known quota-related eligibility information

### Requirement: Retry timing is represented when known
The system SHALL capture a deployment's next eligible time when provider responses or router logic provide enough information to infer it.

#### Scenario: Provider returns retry timing
- **WHEN** a deployment returns rate-limit information that includes retry timing or equivalent reset guidance
- **THEN** the normalized state MUST expose a next-eligible timestamp or duration for strategy use

### Requirement: Availability state updates after request outcomes
The system SHALL update normalized deployment availability state after success, failure, cooldown, and rate-limit events.

#### Scenario: Rate limit updates state
- **WHEN** a deployment returns a rate-limit response
- **THEN** the system MUST update that deployment's availability state before future custom strategy evaluations

### Requirement: Shared state remains consistent in multi-instance mode
The system SHALL store deployment availability state in shared storage when Redis-backed multi-instance routing is enabled.

#### Scenario: Two proxy instances share router state
- **WHEN** one LiteLLM instance observes a deployment entering cooldown or receiving a retry-after signal and Redis-backed routing is enabled
- **THEN** other LiteLLM instances MUST be able to observe the updated normalized state for subsequent routing decisions
