## ADDED Requirements

### Requirement: Operators can select a custom rotation strategy
The system SHALL allow router or proxy configuration to select a custom rotation strategy for a model group.

#### Scenario: Router uses a configured custom strategy
- **WHEN** a model group is configured to use a named custom rotation strategy
- **THEN** the router MUST invoke that strategy during deployment selection for requests targeting the model group

### Requirement: Custom strategies receive request context and eligible candidates
The system SHALL provide each custom rotation strategy with the request context and the set of deployments that remain eligible after router health and pre-call checks.

#### Scenario: Strategy inspects quota-aware request selection inputs
- **WHEN** the router invokes a custom rotation strategy
- **THEN** the strategy MUST receive the requested model group, request metadata, and the currently eligible candidate deployments

### Requirement: Router safeguards remain authoritative
The system SHALL continue to enforce cooldowns, retries, and fallback rules even when a custom rotation strategy is configured.

#### Scenario: Strategy selects a deployment that becomes unavailable
- **WHEN** the selected deployment fails or becomes ineligible during request execution
- **THEN** the router MUST apply its configured retry, cooldown, and fallback behavior instead of terminating custom-routing support entirely

### Requirement: Strategy failures degrade safely
The system SHALL handle custom strategy exceptions without leaving the router in an undefined state.

#### Scenario: Strategy raises an exception
- **WHEN** a custom rotation strategy throws an error during deployment selection
- **THEN** the system MUST emit an actionable error or fall back according to configured router behavior without corrupting deployment state
