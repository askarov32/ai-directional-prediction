const DEMO_TEMPLATE = {
  model: "meshgraphnet",
  medium_id: "",
  scenario: {
    temperature_c: 120,
    pressure_mpa: 35,
    time_ms: 12,
  },
  source: {
    type: "thermal_pulse",
    x: 0.15,
    y: 0.4,
    z: 0.2,
    amplitude: 1,
    frequency_hz: 50,
    direction: [1, 0.15, 0.1],
  },
  probe: {
    x: 0.7,
    y: 0.55,
    z: 0.75,
  },
  domain: {
    type: "rect_3d",
    size: {
      lx: 1,
      ly: 1,
      lz: 1,
    },
    resolution: {
      nx: 128,
      ny: 128,
      nz: 48,
    },
    boundary_conditions: {
      left: "fixed",
      right: "free",
      top: "insulated",
      bottom: "insulated",
      front: "insulated",
      back: "insulated",
    },
  },
};

function cloneTemplate() {
  return structuredClone(DEMO_TEMPLATE);
}

function toRect2d(payload) {
  const next = structuredClone(payload);
  next.domain.type = "rect_2d";
  next.domain.size.lz = 0;
  next.domain.resolution.nz = 1;
  next.source.z = 0;
  next.probe.z = 0;
  next.source.direction[2] = 0;
  return next;
}

function toRect3d(payload) {
  const next = structuredClone(payload);
  next.domain.type = "rect_3d";
  next.domain.size.lz = next.domain.size.lz > 0 ? next.domain.size.lz : 1;
  next.domain.resolution.nz = next.domain.resolution.nz > 1 ? next.domain.resolution.nz : 48;
  next.source.z = next.source.z > 0 ? next.source.z : 0.2;
  next.probe.z = next.probe.z > 0 ? next.probe.z : 0.75;
  if (next.source.direction[2] === 0) {
    next.source.direction[2] = 0.1;
  }
  return next;
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

export function applyModelDomainPolicy(payload, modelOrId) {
  const model = normalizeModelDescriptor(modelOrId);
  if (!model) {
    return payload;
  }

  const supported = Array.isArray(model.supported_domain_types) ? model.supported_domain_types : [];
  const defaultDomainType = model.default_domain_type || payload.domain.type;
  if (!supported.length || supported.includes(payload.domain.type)) {
    return payload;
  }

  if (defaultDomainType === "rect_2d") {
    return toRect2d(payload);
  }
  if (defaultDomainType === "rect_3d") {
    return toRect3d(payload);
  }
  return payload;
}

function readNumber(form, selector) {
  return Number(form.querySelector(selector).value);
}

export function buildDemoPayload(mediumId, modelOrId) {
  const model = normalizeModelDescriptor(modelOrId);
  const basePayload = {
    ...cloneTemplate(),
    medium_id: mediumId || DEMO_TEMPLATE.medium_id,
    model: model?.id || DEMO_TEMPLATE.model,
  };
  return applyModelDomainPolicy(basePayload, model);
}

export function readPayloadFromForm(form) {
  return {
    model: form.querySelector("#model-select").value,
    medium_id: form.querySelector("#medium-select").value,
    scenario: {
      temperature_c: readNumber(form, "#temperature-input"),
      pressure_mpa: readNumber(form, "#pressure-input"),
      time_ms: readNumber(form, "#time-input"),
    },
    source: {
      type: form.querySelector("#source-type-input").value,
      x: readNumber(form, "#source-x-input"),
      y: readNumber(form, "#source-y-input"),
      z: readNumber(form, "#source-z-input"),
      amplitude: readNumber(form, "#source-amplitude-input"),
      frequency_hz: readNumber(form, "#source-frequency-input"),
      direction: [
        readNumber(form, "#direction-x-input"),
        readNumber(form, "#direction-y-input"),
        readNumber(form, "#direction-z-input"),
      ],
    },
    probe: {
      x: readNumber(form, "#probe-x-input"),
      y: readNumber(form, "#probe-y-input"),
      z: readNumber(form, "#probe-z-input"),
    },
    domain: {
      type: form.querySelector("#domain-type-input").value,
      size: {
        lx: readNumber(form, "#domain-lx-input"),
        ly: readNumber(form, "#domain-ly-input"),
        lz: readNumber(form, "#domain-lz-input"),
      },
      resolution: {
        nx: readNumber(form, "#domain-nx-input"),
        ny: readNumber(form, "#domain-ny-input"),
        nz: readNumber(form, "#domain-nz-input"),
      },
      boundary_conditions: {
        left: form.querySelector("#boundary-left-input").value,
        right: form.querySelector("#boundary-right-input").value,
        top: form.querySelector("#boundary-top-input").value,
        bottom: form.querySelector("#boundary-bottom-input").value,
      },
    },
  };
}

export function fillForm(form, payload) {
  form.querySelector("#model-select").value = payload.model;
  form.querySelector("#medium-select").value = payload.medium_id;
  form.querySelector("#temperature-input").value = payload.scenario.temperature_c;
  form.querySelector("#pressure-input").value = payload.scenario.pressure_mpa;
  form.querySelector("#time-input").value = payload.scenario.time_ms;

  form.querySelector("#source-type-input").value = payload.source.type;
  form.querySelector("#source-x-input").value = payload.source.x;
  form.querySelector("#source-y-input").value = payload.source.y;
  form.querySelector("#source-z-input").value = payload.source.z;
  form.querySelector("#source-amplitude-input").value = payload.source.amplitude;
  form.querySelector("#source-frequency-input").value = payload.source.frequency_hz;
  form.querySelector("#direction-x-input").value = payload.source.direction[0];
  form.querySelector("#direction-y-input").value = payload.source.direction[1];
  form.querySelector("#direction-z-input").value = payload.source.direction[2];

  form.querySelector("#probe-x-input").value = payload.probe.x;
  form.querySelector("#probe-y-input").value = payload.probe.y;
  form.querySelector("#probe-z-input").value = payload.probe.z;

  form.querySelector("#domain-type-input").value = payload.domain.type;
  form.querySelector("#domain-lx-input").value = payload.domain.size.lx;
  form.querySelector("#domain-ly-input").value = payload.domain.size.ly;
  form.querySelector("#domain-lz-input").value = payload.domain.size.lz;
  form.querySelector("#domain-nx-input").value = payload.domain.resolution.nx;
  form.querySelector("#domain-ny-input").value = payload.domain.resolution.ny;
  form.querySelector("#domain-nz-input").value = payload.domain.resolution.nz;
  form.querySelector("#boundary-left-input").value = payload.domain.boundary_conditions.left;
  form.querySelector("#boundary-right-input").value = payload.domain.boundary_conditions.right;
  form.querySelector("#boundary-top-input").value = payload.domain.boundary_conditions.top;
  form.querySelector("#boundary-bottom-input").value = payload.domain.boundary_conditions.bottom;
}
