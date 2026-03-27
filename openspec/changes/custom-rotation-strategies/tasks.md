## 1. Strategy Extension Surface

- [ ] 1.1 Define the custom rotation strategy interface and configuration surface for router/proxy use.
- [ ] 1.2 Implement strategy registration, loading, and safe invocation paths.

## 2. Availability State and Routing

- [ ] 2.1 Implement normalized deployment-availability state, including cooldown and quota-related eligibility signals.
- [ ] 2.2 Integrate custom strategy selection into router candidate picking without bypassing existing safeguards.
- [ ] 2.3 Add Redis-backed shared state support for multi-instance strategy behavior.

## 3. Reliability and Observability

- [ ] 3.1 Add tests for strategy-driven routing, retry behavior, and fallback interaction.
- [ ] 3.2 Add tests for rate-limit or retry-after updates feeding normalized availability state.
- [ ] 3.3 Add logs, metrics, and documentation for strategy choice, deployment selection, and failure handling.
