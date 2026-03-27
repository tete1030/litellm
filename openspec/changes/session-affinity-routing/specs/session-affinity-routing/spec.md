## ADDED Requirements

### Requirement: Requests can declare an affinity marker
The system SHALL support explicit request markers that can be converted into an affinity key for routing.

#### Scenario: Request provides a session identifier
- **WHEN** a request includes a supported session-affinity marker
- **THEN** the router MUST derive a stable affinity key from that marker before deployment selection

#### Scenario: Request provides a cache key
- **WHEN** a request includes a supported cache-affinity marker such as a prompt cache key
- **THEN** the router MUST derive a stable affinity key that can be reused by later matching requests

### Requirement: Eligible bindings are preferred
The system SHALL route requests to the deployment currently bound to the affinity key when that deployment remains eligible.

#### Scenario: Existing binding remains healthy
- **WHEN** an affinity key is already bound to a deployment and that deployment passes current eligibility checks
- **THEN** the router MUST prefer the bound deployment over general rotation

### Requirement: Ineligible bindings are rebound safely
The system SHALL select and store a new deployment for an affinity key when the existing bound deployment is no longer eligible.

#### Scenario: Bound deployment enters cooldown
- **WHEN** a request arrives for an affinity key whose bound deployment is in cooldown or otherwise ineligible
- **THEN** the router MUST select a new eligible deployment and MUST update the affinity binding for future matching requests

### Requirement: Affinity coexists with reliability features
The system SHALL preserve retry, fallback, and cooldown behavior for affinity-routed requests.

#### Scenario: Affinity-routed request fails
- **WHEN** a request selected through affinity encounters a retryable failure
- **THEN** the router MUST apply configured retry and fallback behavior according to router policy
