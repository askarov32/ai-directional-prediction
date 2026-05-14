# Model Service Predict Curls

This document captures the current direct `/predict` contracts for the four model services.

Important:

- These are direct model-service requests.
- They are not the same as the public backend contract `POST /api/v1/predictions`.
- The backend enriches the request before forwarding it to the selected model service.
- Localhost ports below come from [docker-compose.yml](/Users/askarovi/Documents/New%20project/docker-compose.yml) and [.env.example](/Users/askarovi/Documents/New%20project/.env.example).

Shared direct request pattern:

- `medium`
- `scenario`
- `source`
- `probe`
- `domain`
- `representation`
- optional `routing_hint`

Example medium used below:

```json
{
  "id": "sandstone_medium",
  "name": "Sandstone (medium)",
  "category": "sedimentary",
  "properties": {
    "rho": 2684.0,
    "porosity_total": 0.34,
    "porosity_effective": 0.27,
    "vp": 6.17,
    "vs": 3.2,
    "thermal_conductivity": 2.5,
    "heat_capacity": 850.0,
    "thermal_expansion": 0.000012
  },
  "ranges": {
    "temperature_c": [-20.0, 300.0],
    "pressure_mpa": [0.1, 1500.0]
  },
  "metadata": {
    "source": "presentation_seed",
    "notes": "Starter preset."
  }
}
```

## 1. PINN service

Source of truth:

- [pinn-service/src/pinn_service/service_schemas.py](/Users/askarovi/Documents/New%20project/pinn-service/src/pinn_service/service_schemas.py)
- [pinn-service/src/pinn_service/service_app.py](/Users/askarovi/Documents/New%20project/pinn-service/src/pinn_service/service_app.py)
- [backend/app/infrastructure/clients/pinn_client.py](/Users/askarovi/Documents/New%20project/backend/app/infrastructure/clients/pinn_client.py)

URLs:

- Localhost: `http://localhost:9003`
- Docker Compose service URL: `http://pinn-service:9000`

### Health

```bash
curl http://localhost:9003/health
```

### Ready

```bash
curl http://localhost:9003/ready
```

Notes:

- Returns `200` only when the checkpoint was loaded successfully.
- Returns `503` when the checkpoint is missing or the configured device is invalid for the container.
- A common local issue is `PINN_DEVICE=cuda` on a machine where container CUDA is not available.

### Predict

```bash
curl -X POST http://localhost:9003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medium": {
      "id": "sandstone_medium",
      "name": "Sandstone (medium)",
      "category": "sedimentary",
      "properties": {
        "rho": 2684.0,
        "porosity_total": 0.34,
        "porosity_effective": 0.27,
        "vp": 6.17,
        "vs": 3.2,
        "thermal_conductivity": 2.5,
        "heat_capacity": 850.0,
        "thermal_expansion": 0.000012
      },
      "ranges": {
        "temperature_c": [-20.0, 300.0],
        "pressure_mpa": [0.1, 1500.0]
      },
      "metadata": {
        "source": "presentation_seed",
        "notes": "Starter preset."
      }
    },
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.40,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.70,
      "y": 0.55,
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    },
    "representation": "physics_informed",
    "routing_hint": "pinn"
  }'
```

### Example successful response

```json
{
  "direction_vector": [0.821, 0.571, 0.0],
  "azimuth_deg": 34.8,
  "elevation_deg": 0.0,
  "magnitude": 0.914,
  "wave_type": "physics_informed",
  "travel_time_ms": 0.111,
  "max_displacement": 0.001327,
  "max_temperature_perturbation": 1.742,
  "model_version": "pinn-baseline@best_model.pth",
  "model_outputs": {
    "feature_names": ["temperature_k", "disp_x", "disp_y", "disp_z"],
    "values": [294.8, 0.00091, 0.00062, 0.0]
  },
  "postprocessed_prediction": {
    "direction_vector": [0.821, 0.571, 0.0],
    "azimuth_deg": 34.8,
    "elevation_deg": 0.0,
    "magnitude": 0.914,
    "wave_type": "physics_informed",
    "travel_time_ms": 0.111,
    "max_displacement": 0.001327,
    "max_temperature_perturbation": 1.742
  },
  "diagnostics": {
    "device": "cpu",
    "smoke_check": {
      "status": "passed",
      "output_feature_count": 4
    }
  }
}
```

### Common errors

- `503 CHECKPOINT_NOT_READY`
- `422 Unprocessable Entity` when the payload violates the strict Pydantic schema

## 2. MGN service

Source of truth:

- [mgn-service/src/service/api.py](/Users/askarovi/Documents/New%20project/mgn-service/src/service/api.py)
- [backend/app/infrastructure/clients/meshgraphnet_client.py](/Users/askarovi/Documents/New%20project/backend/app/infrastructure/clients/meshgraphnet_client.py)

URLs:

- Localhost: `http://localhost:9001`
- Docker Compose service URL: `http://mgn-service:9000`

### Health

```bash
curl http://localhost:9001/health
```

### Ready

```bash
curl http://localhost:9001/ready
```

Notes:

- If both dataset and checkpoint exist, mode is `rollout`.
- If they are missing but `MGN_ALLOW_FALLBACK=true`, `/ready` can still return `200` with mode `fallback`.

### Predict

```bash
curl -X POST http://localhost:9001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medium": {
      "id": "sandstone_medium",
      "name": "Sandstone (medium)",
      "category": "sedimentary",
      "properties": {
        "rho": 2684.0,
        "porosity_total": 0.34,
        "porosity_effective": 0.27,
        "vp": 6.17,
        "vs": 3.2,
        "thermal_conductivity": 2.5,
        "heat_capacity": 850.0,
        "thermal_expansion": 0.000012
      }
    },
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.40,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.70,
      "y": 0.55,
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    },
    "representation": "graph",
    "routing_hint": "meshgraphnet"
  }'
```

### Example successful response

```json
{
  "direction_vector": [0.98387, 0.178885, 0.0],
  "azimuth_deg": 10.3052,
  "elevation_deg": 0.0,
  "magnitude": 1.0,
  "wave_type": "dominant_p",
  "travel_time_ms": 2.49092,
  "max_displacement": 0.00174,
  "max_temperature_perturbation": 1.357143,
  "model_version": "real-meshgraphnet-v1",
  "extra_metrics": {
    "max_von_mises_stress": 12.6,
    "max_velocity": 0.42,
    "risk_flag": 0.0
  }
}
```

### Common errors

- `503` with `detail.message="MeshGraphNet dataset not found"` if fallback is off and dataset is missing
- `503` with `detail.message="MeshGraphNet checkpoint not found"` if fallback is off and checkpoint is missing

## 3. FNO service

Source of truth:

- [fno-service/src/fno_service/api/schemas.py](/Users/askarovi/Documents/New%20project/fno-service/src/fno_service/api/schemas.py)
- [fno-service/src/fno_service/api/routes.py](/Users/askarovi/Documents/New%20project/fno-service/src/fno_service/api/routes.py)
- [fno-service/src/fno_service/inference/predictor.py](/Users/askarovi/Documents/New%20project/fno-service/src/fno_service/inference/predictor.py)
- [backend/app/infrastructure/clients/fno_client.py](/Users/askarovi/Documents/New%20project/backend/app/infrastructure/clients/fno_client.py)

URLs:

- Localhost: `http://localhost:9002`
- Docker Compose service URL: `http://fno-service:9000`

### Health

```bash
curl http://localhost:9002/health
```

### Ready

```bash
curl http://localhost:9002/ready
```

Notes:

- If a checkpoint exists and can be loaded, mode is `checkpoint`.
- If no checkpoint exists but `FNO_ALLOW_FALLBACK=true`, `/ready` still returns `200` with mode `fallback`.
- Current baseline supports only `rect_2d` and `Z=1`.

### Predict

```bash
curl -X POST http://localhost:9002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medium": {
      "id": "sandstone_medium",
      "name": "Sandstone (medium)",
      "category": "sedimentary",
      "properties": {
        "rho": 2684.0,
        "porosity_total": 0.34,
        "porosity_effective": 0.27,
        "vp": 6.17,
        "vs": 3.2,
        "thermal_conductivity": 2.5,
        "heat_capacity": 850.0,
        "thermal_expansion": 0.000012
      }
    },
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.40,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.70,
      "y": 0.55,
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    },
    "representation": "grid",
    "routing_hint": "fno",
    "requested_outputs": ["direction", "field_summary"],
    "grid_policy": "service_default"
  }'
```

### Example successful response

Checkpoint mode:

```json
{
  "prediction": {
    "direction_vector": [0.981194, 0.193022, 0.0],
    "azimuth_deg": 11.1292,
    "elevation_deg": 0.0,
    "magnitude": 0.24811,
    "wave_type": "fno_checkpoint_inference",
    "travel_time_ms": 2.363967
  },
  "field_summary": {
    "max_displacement": 0.00145952,
    "max_temperature_perturbation": 1.2
  },
  "model_version": "fno-baseline@best_model.pth",
  "diagnostics": {
    "checkpoint_loaded": true,
    "device": "cpu",
    "mode": "checkpoint"
  }
}
```

Fallback mode:

```json
{
  "prediction": {
    "direction_vector": [0.981194, 0.193022, 0.0],
    "azimuth_deg": 11.1292,
    "elevation_deg": 0.0,
    "magnitude": 1.0,
    "wave_type": "fno_skeleton_fallback",
    "travel_time_ms": 2.363967
  },
  "field_summary": {
    "max_displacement": 0.00145952,
    "max_temperature_perturbation": 1.2
  },
  "model_version": "fno-skeleton-fallback-v0",
  "diagnostics": {
    "mode": "fallback",
    "note": "Fallback response because no FNO checkpoint is configured."
  }
}
```

### Common errors

- `503 CHECKPOINT_NOT_READY`
- `400 UNSUPPORTED_DOMAIN` for unsupported geometry like `rect_3d` or `nz > 1`
- `500 MODEL_LOAD_FAILED`
- `500 NON_FINITE_MODEL_OUTPUT`

## 4. Transformer service

Source of truth:

- [transformer-service/src/transformer_service/service_schemas.py](/Users/askarovi/Documents/New%20project/transformer-service/src/transformer_service/service_schemas.py)
- [transformer-service/src/transformer_service/service_app.py](/Users/askarovi/Documents/New%20project/transformer-service/src/transformer_service/service_app.py)
- [transformer-service/src/transformer_service/inference_service.py](/Users/askarovi/Documents/New%20project/transformer-service/src/transformer_service/inference_service.py)
- [backend/app/infrastructure/clients/transformer_client.py](/Users/askarovi/Documents/New%20project/backend/app/infrastructure/clients/transformer_client.py)

URLs:

- Localhost: `http://localhost:9004`
- Docker Compose service URL: `http://transformer-service:9000`

### Health

```bash
curl http://localhost:9004/health
```

### Ready

```bash
curl http://localhost:9004/ready
```

Notes:

- Returns `200` only when the Transformer checkpoint was loaded and passed the smoke check.
- Returns `503` when the checkpoint is absent or invalid.

### Predict

```bash
curl -X POST http://localhost:9004/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medium": {
      "id": "sandstone_medium",
      "name": "Sandstone (medium)",
      "category": "sedimentary",
      "properties": {
        "rho": 2684.0,
        "porosity_total": 0.34,
        "porosity_effective": 0.27,
        "vp": 6.17,
        "vs": 3.2,
        "thermal_conductivity": 2.5,
        "heat_capacity": 850.0,
        "thermal_expansion": 0.000012
      },
      "ranges": {
        "temperature_c": [-20.0, 300.0],
        "pressure_mpa": [0.1, 1500.0]
      },
      "metadata": {
        "source": "presentation_seed",
        "notes": "Starter preset."
      }
    },
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.40,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.70,
      "y": 0.55,
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    },
    "representation": "tokenset",
    "routing_hint": "transformer"
  }'
```

### Example successful response

```json
{
  "direction_vector": [0.889121, 0.457671, 0.0],
  "azimuth_deg": 27.235,
  "elevation_deg": 0.0,
  "magnitude": 0.318,
  "wave_type": "tokenset",
  "travel_time_ms": 0.132441,
  "max_displacement": 0.000212,
  "max_temperature_perturbation": 1.1034,
  "model_version": "oformer-baseline@best_model.pth",
  "model_outputs": {
    "feature_names": ["temperature_k", "disp_x", "disp_y", "disp_z"],
    "final_step_values": [294.1, 0.00014, 0.00009, 0.0]
  },
  "postprocessed_prediction": {
    "direction_vector": [0.889121, 0.457671, 0.0],
    "azimuth_deg": 27.235,
    "elevation_deg": 0.0,
    "magnitude": 0.318,
    "wave_type": "tokenset",
    "travel_time_ms": 0.132441,
    "max_displacement": 0.000212,
    "max_temperature_perturbation": 1.1034
  },
  "diagnostics": {
    "device": "cpu",
    "rollout_steps": 50,
    "smoke_check": {
      "status": "passed"
    }
  }
}
```

### Common errors

- `503 CHECKPOINT_NOT_READY`
- `500 INFERENCE_FAILURE`
- `422 Unprocessable Entity` when the request violates the strict schema

## Response-shape summary

The direct response schemas are not uniform.

Flat postprocessed response:

- `pinn-service`
- `mgn-service`
- `transformer-service`

Nested response:

- `fno-service`
  - `prediction`
  - `field_summary`
  - `model_version`
  - `diagnostics`

This is why the experiment runner should normalize all direct service responses into one canonical record before saving charts or metrics.
