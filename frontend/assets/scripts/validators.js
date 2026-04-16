export function formatRange(range, unit) {
  if (!range || range.length !== 2) {
    return "";
  }
  return `Allowed range: ${range[0]} to ${range[1]} ${unit}`.trim();
}

function within(value, min, max) {
  return value >= min && value <= max;
}

function pointInBounds(point, size) {
  return within(point.x, 0, size.lx) && within(point.y, 0, size.ly) && within(point.z, 0, size.lz);
}

export function validatePayload(payload, medium) {
  const errors = {};

  if (!medium) {
    errors["medium"] = "Select a geological medium.";
    return errors;
  }

  const [minTemp, maxTemp] = medium.ranges.temperature_c;
  if (!within(payload.scenario.temperature_c, minTemp, maxTemp)) {
    errors["scenario.temperature_c"] = formatRange(medium.ranges.temperature_c, "°C");
  }

  const [minPressure, maxPressure] = medium.ranges.pressure_mpa;
  if (!within(payload.scenario.pressure_mpa, minPressure, maxPressure)) {
    errors["scenario.pressure_mpa"] = formatRange(medium.ranges.pressure_mpa, "MPa");
  }

  const { size, resolution, type } = payload.domain;
  if (size.lx <= 0 || size.ly <= 0 || size.lz < 0) {
    errors["domain.size"] = "Domain dimensions must be positive and Lz cannot be negative.";
  }

  if (resolution.nx < 2 || resolution.ny < 2 || resolution.nz < 1) {
    errors["domain.resolution"] = "Resolution must be at least 2 x 2 x 1.";
  }

  if (type === "rect_2d" && (size.lz !== 0 || resolution.nz !== 1)) {
    errors["domain.resolution"] = "rect_2d requires Lz = 0 and Nz = 1.";
  }

  if (!pointInBounds(payload.source, size)) {
    errors["source.coordinates"] = "Source coordinates must stay inside the domain.";
  }

  if (!pointInBounds(payload.probe, size)) {
    errors["probe.coordinates"] = "Probe coordinates must stay inside the domain.";
  }

  const directionMagnitude = Math.sqrt(payload.source.direction.reduce((sum, value) => sum + value * value, 0));
  if (directionMagnitude === 0) {
    errors["source.direction"] = "Direction vector magnitude must be greater than zero.";
  }

  return errors;
}
