## 1. Profile Configuration

- [x] 1.1 Add ChatGPT auth profile configuration and backward-compatible default-profile resolution.
- [x] 1.2 Implement a profile registry that resolves token storage locations and caches authenticators by profile.

## 2. Provider Integration

- [x] 2.1 Thread deployment-level ChatGPT auth profile selection through all `chatgpt/*` request paths.
- [x] 2.2 Add per-profile refresh locking and atomic auth-state persistence.
- [x] 2.3 Surface clear validation and runtime errors for unknown or unusable profiles.

## 3. Verification and Rollout

- [x] 3.1 Add tests for multiple ChatGPT deployments using different profiles and for legacy single-profile configs.
- [x] 3.2 Add concurrency tests for simultaneous refreshes on the same profile.
- [x] 3.3 Update operator docs with profile setup, migration guidance, and troubleshooting steps.
