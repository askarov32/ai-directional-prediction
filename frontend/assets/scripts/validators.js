export function formatRange(range, unit) {
  if (!range || range.length !== 2) {
    return "";
  }
  return `Allowed range: ${range[0]} to ${range[1]} ${unit}`.trim();
}

function finite(value) {
  return Number.isFinite(Number(value));
}

function withinUnitPlane(point) {
  return finite(point.x_m) && finite(point.y_m) && point.x_m >= 0 && point.x_m <= 1 && point.y_m >= 0 && point.y_m <= 1;
}

export function validatePayload(payload, medium, model = null) {
  const errors = {};

  if (!medium) {
    errors.medium = "Select a geological medium.";
    return errors;
  }

  if (!model) {
    errors.model = "Select a model route.";
  }

  if (medium.thermoelastic_supported === false) {
    errors.medium = "This medium is not available for the current prototype prediction workflow.";
  }

  if (payload.schema_version !== "2.0") {
    errors.schema = "Frontend requests must use API Contract v2.";
  }

  if (payload.geometry?.dimension !== 2) {
    errors["geometry.source"] = "The demo uses planar source and probe coordinates.";
  }

  const source = payload.geometry?.source || {};
  const probe = payload.geometry?.probe || {};

  if (!withinUnitPlane(source)) {
    errors["geometry.source"] = "Source coordinates must stay within 0 to 1 m.";
  }

  if (!withinUnitPlane(probe)) {
    errors["geometry.probe"] = "Probe coordinates must stay within 0 to 1 m.";
  }

  if (
    finite(source.x_m) &&
    finite(source.y_m) &&
    finite(probe.x_m) &&
    finite(probe.y_m) &&
    source.x_m === probe.x_m &&
    source.y_m === probe.y_m
  ) {
    errors["geometry.probe"] = "Probe must be different from the source.";
  }

  if (!finite(payload.observation?.time_s) || payload.observation.time_s <= 0) {
    errors["observation.time_s"] = "Observation time must be greater than zero.";
  }

  return errors;
}
