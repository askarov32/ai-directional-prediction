# Demo Limitations

## What Is Production-Style

The following parts are implemented in a production-oriented MVP style:

- FastAPI backend with versioned `/api/v1` routes;
- clean use case and routing layers;
- strict request validation;
- strict remote model response validation;
- typed remote model errors;
- readiness endpoints;
- request id propagation;
- Docker Compose local stack;
- non-root Python service containers;
- nginx proxy for frontend-to-backend API calls;
- security headers for static frontend responses;
- baseline backend and PINN tests.

## What Is Demo-Only

The following parts are intentionally demo/MVP level:

- MeshGraphNet service is mocked;
- FNO service is mocked;
- medium catalog values are starter presets;
- PINN baseline is not yet independently validated;
- frontend visualization is illustrative;
- model comparison is not yet a statistically rigorous benchmark;
- no authentication, authorization, rate limiting, or audit logging;
- no production monitoring/tracing/metrics backend.

## Scientific Claims To Avoid

Do not claim:

- that this predicts real geological wave propagation with validated accuracy;
- that MeshGraphNet or FNO are trained and scientifically evaluated in this repository;
- that the PINN baseline solves the full coupled thermoelastic PDE in real time;
- that current rock presets are final laboratory/reference values;
- that output is safe for engineering decisions.

## Safe Demo Claims

You can claim:

- the system supports a unified prediction request contract;
- the backend resolves geological media from JSON presets;
- the backend routes requests to MeshGraphNet, FNO, and PINN services;
- remote model responses are validated and normalized;
- local Docker setup is one-command demo-ready;
- PINN service loads a real checkpoint and exposes readiness diagnostics;
- MeshGraphNet/FNO can be replaced by real model hosts via environment variables.

## Current Validation Level

Engineering validation:

- backend tests pass;
- PINN contract tests pass;
- Docker Compose stack builds and starts;
- readiness endpoints verify service availability;
- sample prediction works through frontend nginx proxy.

Scientific validation:

- not complete;
- requires held-out COMSOL/experimental cases;
- requires directional angular error metrics;
- requires documented train/validation/test split;
- requires sensitivity and ablation analysis.

## Suggested Next Scientific Work

- Prepare separate validation data from COMSOL or experiments.
- Report angular error for direction prediction.
- Report travel-time error.
- Compare PINN, MeshGraphNet, and FNO on the same fixed scenarios.
- Replace mock MeshGraphNet and FNO services with trained services.
- Validate medium presets against references or lab measurements.
