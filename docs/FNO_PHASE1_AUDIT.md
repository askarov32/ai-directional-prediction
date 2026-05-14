# FNO Phase 1 Audit

This document records the Phase 1 repository audit for the FNO integration roadmap.

Roadmap source:

```text
docs/FNO_INTEGRATION_ROADMAP.md
```

## Current Service State

Current Docker Compose services:

```text
frontend
backend
mgn-service
mock-fno
mock-transformer
pinn-service
transformer-service
```

Current FNO route:

```text
backend MODEL_FNO_URL -> http://mock-fno:9000
```

Conclusion:

```text
FNO is still mocked in the default Docker stack.
```

## Backend FNO Routing

The backend already has the FNO integration points:

```text
backend/app/domain/enums/model_type.py
backend/app/infrastructure/clients/fno_client.py
backend/app/domain/services/prediction_router.py
backend/app/api/dependencies.py
backend/app/core/config.py
```

Confirmed model enum values:

```text
meshgraphnet
fno
transformer
pinn
```

Confirmed FNO client behavior:

```json
{
  "representation": "grid",
  "routing_hint": "fno"
}
```

Conclusion:

```text
The backend routing architecture is already suitable for a real FNO service.
No router rewrite is needed for the first FNO implementation phases.
```

## Current FNO Service State

Current repository state:

```text
fno-service/ does not exist yet.
```

Conclusion:

```text
Phase 2 should create the fno-service skeleton before any neural network implementation.
```

## Mock FNO State

Current mock FNO files:

```text
mock-services/main.py
mock-services/common/predictor.py
```

Current Docker Compose service:

```text
mock-fno
```

Conclusion:

```text
mock-fno should stay available as an optional demo fallback later, but the default route should move to fno-service after the real service skeleton is ready.
```

## Dataset State

PINN artifacts currently exist:

```text
pinn-service/artifacts/demo/structured_dataset.npz
pinn-service/artifacts/demo/training_samples.npz
pinn-service/artifacts/rod_experiments/*/structured_dataset.npz
pinn-service/artifacts/rod_experiments/*/training_samples.npz
pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz
pinn-service/artifacts/rod_experiments/manifest.json
```

MGN universal formatter exists:

```text
mgn-service/scripts/reformat_dataset.py
mgn-service/src/data/universal_formatter.py
```

Current observation:

```text
No ready processed/fno grid artifacts were found in the current local tree during this audit.
```

Phase 3 decision:

```text
Prefer the universal formatter FNO output when available.
Keep a fallback path from PINN structured_dataset.npz to an FNO grid for local MVP continuity.
```

Phase 3 implementation note:

```text
The fno-service now includes an FNO grid tensor loader and a fallback converter from PINN structured_dataset.npz.
The converter writes the same [T,C,Z,Y,X] layout used by mgn-service/src/data/universal_formatter.py.
```

## Transformer Note

The repository currently contains:

```text
transformer-service/
```

Docker Compose also includes:

```text
transformer-service
mock-transformer
```

Conclusion:

```text
Transformer wiring is outside the FNO integration scope.
Do not mix transformer cleanup with FNO Phase 2.
```

## Phase 2 Starting Point

Recommended next step:

```text
Implement only Phase 2 and minimal API tests:
- create fno-service skeleton;
- add Dockerfile and requirements;
- add FastAPI /health, /ready, /predict placeholders;
- return CHECKPOINT_NOT_READY from /predict when no checkpoint exists;
- add tests for API behavior;
- do not implement FNO neural network yet.
```
