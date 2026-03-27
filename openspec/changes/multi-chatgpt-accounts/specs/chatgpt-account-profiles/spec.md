## ADDED Requirements

### Requirement: Deployments can bind to named ChatGPT auth profiles
The system SHALL allow each `chatgpt/*` deployment to select a named ChatGPT auth profile so multiple OAuth accounts can coexist within the same LiteLLM instance.

#### Scenario: Two deployments use different accounts
- **WHEN** two deployments for the same logical model are configured with different ChatGPT auth profile names
- **THEN** the router MUST treat them as separate deployments that authenticate with independent ChatGPT OAuth state

#### Scenario: Existing configuration omits a profile
- **WHEN** a `chatgpt/*` deployment does not specify a ChatGPT auth profile
- **THEN** the system MUST resolve the deployment to a backward-compatible default profile

### Requirement: ChatGPT auth state is isolated per profile
The system SHALL keep token files, cached authenticators, and refresh state isolated by resolved ChatGPT auth profile.

#### Scenario: One profile refreshes without affecting another
- **WHEN** profile `account-a` refreshes its access token while profile `account-b` is also configured
- **THEN** the refresh MUST update only `account-a` state and MUST NOT modify `account-b` token state

### Requirement: Profile refresh writes are concurrency-safe
The system SHALL apply per-profile synchronization so concurrent requests do not corrupt ChatGPT auth state during token refresh.

#### Scenario: Concurrent refresh on one profile
- **WHEN** multiple requests detect an expired token for the same ChatGPT auth profile at nearly the same time
- **THEN** the system MUST serialize profile refresh persistence and MUST leave the stored auth state in a valid, readable form

### Requirement: Invalid profile selection fails clearly
The system SHALL reject deployments or requests that reference an undefined or unusable ChatGPT auth profile with an actionable error.

#### Scenario: Deployment references an unknown profile
- **WHEN** a deployment is configured with a ChatGPT auth profile name that cannot be resolved
- **THEN** the system MUST fail validation or request initialization with an error that identifies the missing profile name
