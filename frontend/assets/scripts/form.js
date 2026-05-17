const DEMO_TEMPLATE = {
  schema_version: "2.0",
  model: "pinn",
  medium_id: "",
  geometry: {
    dimension: 2,
    source: { x_m: 0.15, y_m: 0.4 },
    probe: { x_m: 0.7, y_m: 0.55 },
  },
  observation: {
    time_s: 0.012,
  },
  scenario: {
    thermal_source_type: "point",
    mechanical_constraint: "free",
    boundary_condition_type: "prototype_simplified",
  },
};

function cloneTemplate() {
  return structuredClone(DEMO_TEMPLATE);
}

function normalizeModelDescriptor(modelOrId) {
  if (!modelOrId) {
    return null;
  }
  if (typeof modelOrId === "string") {
    return { id: modelOrId };
  }
  return modelOrId;
}

function readNumber(form, selector) {
  return Number(form.querySelector(selector).value);
}

export function buildDemoPayload(mediumId, modelOrId) {
  const model = normalizeModelDescriptor(modelOrId);
  return {
    ...cloneTemplate(),
    medium_id: mediumId || DEMO_TEMPLATE.medium_id,
    model: model?.id || DEMO_TEMPLATE.model,
  };
}

export function readPayloadFromForm(form) {
  return {
    schema_version: "2.0",
    model: form.querySelector("#model-select").value,
    medium_id: form.querySelector("#medium-select").value,
    geometry: {
      dimension: 2,
      source: {
        x_m: readNumber(form, "#source-x-input"),
        y_m: readNumber(form, "#source-y-input"),
      },
      probe: {
        x_m: readNumber(form, "#probe-x-input"),
        y_m: readNumber(form, "#probe-y-input"),
      },
    },
    observation: {
      time_s: readNumber(form, "#time-input"),
    },
    scenario: {
      thermal_source_type: "point",
      mechanical_constraint: "free",
      boundary_condition_type: "prototype_simplified",
    },
  };
}

export function fillForm(form, payload) {
  form.querySelector("#model-select").value = payload.model;
  form.querySelector("#medium-select").value = payload.medium_id;
  form.querySelector("#source-x-input").value = payload.geometry.source.x_m;
  form.querySelector("#source-y-input").value = payload.geometry.source.y_m;
  form.querySelector("#probe-x-input").value = payload.geometry.probe.x_m;
  form.querySelector("#probe-y-input").value = payload.geometry.probe.y_m;
  form.querySelector("#time-input").value = payload.observation.time_s;
}
