function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function finiteNumber(value, fallback = NaN) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

function formatCoordinate(value) {
  return formatNumber(value, 3);
}

function toTitleCase(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

const COLORS = {
  domainFill: "#172033",
  domainStroke: "rgba(148, 163, 184, 0.88)",
  grid: "rgba(148, 163, 184, 0.22)",
  guide: "rgba(226, 232, 240, 0.36)",
  sourceFill: "#f59e0b",
  sourceStroke: "#fde68a",
  probeFill: "#38bdf8",
  probeStroke: "#bae6fd",
  arrow: "#f8fafc",
  arrowGlow: "rgba(248, 250, 252, 0.18)",
  angle: "#c4b5fd",
  label: "#f8fafc",
  meta: "#94a3b8",
  tooltipFill: "rgba(15, 23, 42, 0.94)",
  tooltipStroke: "rgba(148, 163, 184, 0.42)",
};

const CHANNEL_LABELS = {
  temperature: "Temperature",
  temperature_k: "Temperature",
  temp: "Temperature",
  temp_k: "Temperature",
  temperature_perturbation: "Temperature perturbation",
  temperature_perturbation_k: "Temperature perturbation",
  displacement_magnitude: "Displacement magnitude",
  displacement_magnitude_m: "Displacement magnitude",
  disp_magnitude: "Displacement magnitude",
  magnitude_m: "Displacement magnitude",
};

const CHANNEL_UNITS = {
  temperature: "K",
  temperature_k: "K",
  temp: "K",
  temp_k: "K",
  temperature_perturbation: "K",
  temperature_perturbation_k: "K",
  displacement_magnitude: "m",
  displacement_magnitude_m: "m",
  disp_magnitude: "m",
  magnitude_m: "m",
};

function buildBoxGeometry() {
  return {
    width: 1000,
    height: 620,
    marginLeft: 90,
    marginRight: 70,
    marginTop: 105,
    marginBottom: 70,
  };
}

function projectPoint(point, box) {
  const innerWidth = box.width - box.marginLeft - box.marginRight;
  const innerHeight = box.height - box.marginTop - box.marginBottom;
  return {
    x: box.marginLeft + clamp(point.x, 0, 1) * innerWidth,
    y: box.marginTop + (1 - clamp(point.y, 0, 1)) * innerHeight,
  };
}

function readPoint(point, fallback = { x: 0, y: 0 }) {
  return {
    x: finiteNumber(point?.x_m ?? point?.x, fallback.x),
    y: finiteNumber(point?.y_m ?? point?.y, fallback.y),
  };
}

function buildFallbackGeometry(payload) {
  const source = readPoint(payload?.geometry?.source ?? payload?.source, { x: 0.15, y: 0.4 });
  const probe = readPoint(payload?.geometry?.probe ?? payload?.probe, { x: 0.7, y: 0.55 });
  return {
    dimension: 2,
    source: { x_m: source.x, y_m: source.y },
    probe: { x_m: probe.x, y_m: probe.y },
  };
}

function computeDistance(source, probe) {
  const dx = probe.x - source.x;
  const dy = probe.y - source.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function computeAzimuth(source, probe) {
  return Math.atan2(probe.y - source.y, probe.x - source.x) * (180 / Math.PI);
}

function readDistance(geometry, directionalResponse, source, probe) {
  const geometryDistance = finiteNumber(geometry?.distance_m);
  if (Number.isFinite(geometryDistance)) {
    return geometryDistance;
  }

  const directionalDistance = finiteNumber(directionalResponse?.distance_m);
  if (Number.isFinite(directionalDistance)) {
    return directionalDistance;
  }

  return computeDistance(source, probe);
}

function readAzimuth(geometry, directionalResponse, source, probe) {
  const geometryAzimuth = finiteNumber(geometry?.azimuth_deg);
  if (Number.isFinite(geometryAzimuth)) {
    return geometryAzimuth;
  }

  const directionalAzimuth = finiteNumber(directionalResponse?.azimuth_deg);
  if (Number.isFinite(directionalAzimuth)) {
    return directionalAzimuth;
  }

  return computeAzimuth(source, probe);
}

function readUnitDirection(geometry, source, probe) {
  const unit = geometry?.unit_direction;
  const ux = finiteNumber(unit?.x);
  const uy = finiteNumber(unit?.y);
  if (Number.isFinite(ux) && Number.isFinite(uy)) {
    return { x: ux, y: uy };
  }

  const vector = geometry?.propagation_vector_m;
  const dx = finiteNumber(vector?.dx);
  const dy = finiteNumber(vector?.dy);
  if (Number.isFinite(dx) && Number.isFinite(dy)) {
    const length = Math.sqrt(dx * dx + dy * dy) || 1;
    return { x: dx / length, y: dy / length };
  }

  const fallbackDx = probe.x - source.x;
  const fallbackDy = probe.y - source.y;
  const fallbackLength = Math.sqrt(fallbackDx * fallbackDx + fallbackDy * fallbackDy) || 1;
  return { x: fallbackDx / fallbackLength, y: fallbackDy / fallbackLength };
}

function pointAt(sourcePoint, probePoint, fraction) {
  return {
    x: sourcePoint.x + (probePoint.x - sourcePoint.x) * fraction,
    y: sourcePoint.y + (probePoint.y - sourcePoint.y) * fraction,
  };
}

function buildGridLines(box) {
  const lines = [];
  const steps = 5;
  for (let index = 1; index < steps; index += 1) {
    const t = index / steps;
    const verticalStart = projectPoint({ x: t, y: 0 }, box);
    const verticalEnd = projectPoint({ x: t, y: 1 }, box);
    const horizontalStart = projectPoint({ x: 0, y: t }, box);
    const horizontalEnd = projectPoint({ x: 1, y: t }, box);
    lines.push(
      `<line x1="${verticalStart.x}" y1="${verticalStart.y}" x2="${verticalEnd.x}" y2="${verticalEnd.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
    );
    lines.push(
      `<line x1="${horizontalStart.x}" y1="${horizontalStart.y}" x2="${horizontalEnd.x}" y2="${horizontalEnd.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
    );
  }
  return lines.join("");
}

function buildDomainMarkup(box) {
  const bottomLeft = projectPoint({ x: 0, y: 0 }, box);
  const bottomRight = projectPoint({ x: 1, y: 0 }, box);
  const topRight = projectPoint({ x: 1, y: 1 }, box);
  const topLeft = projectPoint({ x: 0, y: 1 }, box);
  return `
    <polygon
      points="${bottomLeft.x},${bottomLeft.y} ${bottomRight.x},${bottomRight.y} ${topRight.x},${topRight.y} ${topLeft.x},${topLeft.y}"
      fill="${COLORS.domainFill}" stroke="${COLORS.domainStroke}" stroke-width="2.8" />
  `;
}

function buildTooltipMarkup(point, lines, align = "right") {
  const width = 214;
  const height = 62;
  const offsetX = align === "left" ? -width - 18 : 22;
  const offsetY = -78;
  const x = clamp(point.x + offsetX, 18, 1000 - width - 18);
  const y = clamp(point.y + offsetY, 18, 620 - height - 18);

  return `
    <g class="geometry-tooltip" aria-hidden="true">
      <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="14" fill="${COLORS.tooltipFill}" stroke="${COLORS.tooltipStroke}" />
      <text x="${x + 16}" y="${y + 25}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="13" font-weight="700" fill="${COLORS.label}">
        ${escapeHtml(lines[0])}
      </text>
      <text x="${x + 16}" y="${y + 45}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">
        ${escapeHtml(lines[1])}
      </text>
    </g>
  `;
}

function buildPointMarkup({ point, label, coordinates, fill, stroke, labelOffsetX, labelOffsetY, tooltipAlign }) {
  const tooltip = `${label}: ${coordinates}`;
  return `
    <g class="geometry-point" tabindex="0" role="img" aria-label="${escapeHtml(tooltip)}">
      <title>${escapeHtml(tooltip)}</title>
      <circle cx="${point.x}" cy="${point.y}" r="18" fill="${fill}" stroke="${stroke}" stroke-width="4" />
      <circle cx="${point.x}" cy="${point.y}" r="28" fill="transparent" />
      <text x="${point.x + labelOffsetX}" y="${point.y + labelOffsetY}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" fill="${COLORS.label}">${escapeHtml(label)}</text>
      <text x="${point.x + labelOffsetX}" y="${point.y + labelOffsetY + 18}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">${escapeHtml(coordinates)}</text>
      ${buildTooltipMarkup(point, [label, coordinates], tooltipAlign)}
    </g>
  `;
}

function buildPropagationMarkup({ sourcePoint, probePoint, distanceM, azimuthDeg }) {
  const mid = pointAt(sourcePoint, probePoint, 0.5);
  const flowPoint = pointAt(sourcePoint, probePoint, 0.68);
  const labelY = mid.y - 24;
  const azimuthY = mid.y + 28;
  const labelX = clamp(mid.x - 96, 100, 820);
  const distanceLabel = `Distance ${formatNumber(distanceM, 3)} m`;
  const azimuthLabel = `Azimuth ${formatNumber(azimuthDeg, 1)}°`;

  return `
    <line
      x1="${sourcePoint.x}" y1="${sourcePoint.y}"
      x2="${probePoint.x}" y2="${probePoint.y}"
      stroke="${COLORS.arrowGlow}" stroke-width="14" stroke-linecap="round" />
    <line
      class="viz-arrow is-active"
      x1="${sourcePoint.x}" y1="${sourcePoint.y}"
      x2="${probePoint.x}" y2="${probePoint.y}"
      stroke="${COLORS.arrow}" stroke-width="5.5" stroke-linecap="round" fill="none"
      marker-end="url(#arrowhead)" />
    <circle class="viz-flow-dot" cx="${flowPoint.x}" cy="${flowPoint.y}" r="5.5" fill="${COLORS.arrow}" opacity="0.92" />
    <g class="geometry-label">
      <rect x="${labelX - 14}" y="${labelY - 24}" width="222" height="74" rx="16" fill="rgba(15, 23, 42, 0.72)" stroke="rgba(148, 163, 184, 0.26)" />
      <text x="${labelX}" y="${labelY}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.label}">${distanceLabel}</text>
      <text x="${labelX}" y="${azimuthY}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.angle}">${azimuthLabel}</text>
    </g>
  `;
}

function buildUnitDirectionMarkup(box, source, unitDirection) {
  const origin = projectPoint({ x: 0.08, y: 0.12 }, box);
  const length = 74;
  const end = {
    x: origin.x + unitDirection.x * length,
    y: origin.y - unitDirection.y * length,
  };

  return `
    <g class="unit-direction-key">
      <text x="${origin.x - 2}" y="${origin.y + 48}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="12" fill="${COLORS.meta}">
        Unit direction
      </text>
      <line
        x1="${origin.x}" y1="${origin.y}"
        x2="${end.x}" y2="${end.y}"
        stroke="${COLORS.angle}" stroke-width="3.4" stroke-linecap="round"
        marker-end="url(#anglehead)" />
      <text x="${origin.x + 88}" y="${origin.y + 4}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">
        [${formatNumber(unitDirection.x, 2)}, ${formatNumber(unitDirection.y, 2)}]
      </text>
      <title>Unit direction from source (${formatCoordinate(source.x)}, ${formatCoordinate(source.y)}) toward the probe.</title>
    </g>
  `;
}

function buildAxisLabels(box) {
  const xLabelX = box.width - box.marginRight + 10;
  const xLabelY = box.height - box.marginBottom + 28;
  const yLabelX = box.marginLeft - 55;
  const yLabelY = box.marginTop - 12;
  const xLabel = `<text x="${xLabelX}" y="${xLabelY}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}" text-anchor="end">x · 1.00 m</text>`;
  const yLabel = `<text x="${yLabelX}" y="${yLabelY}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">y · 1.00 m</text>`;
  return `${xLabel}${yLabel}`;
}

export function renderGeometryDiagram(svg, geometry, directionalResponse, options = {}) {
  if (!svg || !geometry) {
    return;
  }

  const box = buildBoxGeometry();
  const source = readPoint(geometry.source, { x: 0.15, y: 0.4 });
  const probe = readPoint(geometry.probe, { x: 0.7, y: 0.55 });
  const sourcePoint = projectPoint(source, box);
  const probePoint = projectPoint(probe, box);
  const distanceM = readDistance(geometry, directionalResponse, source, probe);
  const azimuthDeg = readAzimuth(geometry, directionalResponse, source, probe);
  const unitDirection = readUnitDirection(geometry, source, probe);
  const modelLabel = options.modelLabel || "No model selected";
  const sourceCoordinates = `x=${formatCoordinate(source.x)}, y=${formatCoordinate(source.y)}`;
  const probeCoordinates = `x=${formatCoordinate(probe.x)}, y=${formatCoordinate(probe.y)}`;

  svg.innerHTML = `
    <defs>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${COLORS.arrow}"></path>
      </marker>
      <marker id="anglehead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${COLORS.angle}"></path>
      </marker>
    </defs>
    ${buildDomainMarkup(box)}
    ${buildGridLines(box)}
    <line x1="${sourcePoint.x}" y1="${sourcePoint.y}" x2="${probePoint.x}" y2="${sourcePoint.y}" stroke="${COLORS.guide}" stroke-width="1.2" stroke-dasharray="5 7" />
    <line x1="${probePoint.x}" y1="${sourcePoint.y}" x2="${probePoint.x}" y2="${probePoint.y}" stroke="${COLORS.guide}" stroke-width="1.2" stroke-dasharray="5 7" />
    ${buildPropagationMarkup({ sourcePoint, probePoint, distanceM, azimuthDeg })}
    ${buildPointMarkup({
      point: sourcePoint,
      label: "Source",
      fill: COLORS.sourceFill,
      stroke: COLORS.sourceStroke,
      coordinates: sourceCoordinates,
      labelOffsetX: 24,
      labelOffsetY: -18,
      tooltipAlign: source.x > 0.72 ? "left" : "right",
    })}
    ${buildPointMarkup({
      point: probePoint,
      label: "Probe",
      fill: COLORS.probeFill,
      stroke: COLORS.probeStroke,
      coordinates: probeCoordinates,
      labelOffsetX: 24,
      labelOffsetY: -18,
      tooltipAlign: probe.x > 0.72 ? "left" : "right",
    })}
    ${buildUnitDirectionMarkup(box, source, unitDirection)}
    <text x="${box.marginLeft}" y="45" font-family="Avenir Next, Segoe UI, sans-serif" font-size="16" font-weight="600" fill="${COLORS.label}">
      ${escapeHtml(modelLabel)}
    </text>
    <text x="${box.marginLeft}" y="70" font-family="Avenir Next, Segoe UI, sans-serif" font-size="13" fill="${COLORS.meta}">
      Fixed 1.00 m x 1.00 m domain
    </text>
    <text x="${box.marginLeft}" y="${box.height - 18}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="12.5" fill="${COLORS.meta}">
      Source-to-probe geometry for the normalized prediction response.
    </text>
    ${buildAxisLabels(box)}
  `;
}

export function renderDomain(svg, payload, response, modelLabel) {
  const fallbackGeometry = buildFallbackGeometry(payload);
  const geometry = response?.geometry || fallbackGeometry;
  const directionalResponse = response?.prediction?.directional_response || null;
  renderGeometryDiagram(svg, geometry, directionalResponse, { modelLabel });
}

function matrixFromData(data, widthHint, heightHint) {
  if (!Array.isArray(data) || data.length === 0) {
    return null;
  }

  if (Array.isArray(data[0])) {
    const matrix = data
      .map((row) => (Array.isArray(row) ? row.map((value) => finiteNumber(value)) : []))
      .filter((row) => row.length > 0);
    return matrix.length > 0 ? matrix : null;
  }

  let width = Math.trunc(finiteNumber(widthHint, 0));
  let height = Math.trunc(finiteNumber(heightHint, 0));

  if (!width && !height) {
    const side = Math.sqrt(data.length);
    if (Number.isInteger(side)) {
      width = side;
      height = side;
    } else {
      width = data.length;
      height = 1;
    }
  } else if (width && !height) {
    height = Math.ceil(data.length / width);
  } else if (!width && height) {
    width = Math.ceil(data.length / height);
  }

  if (!width || !height || width * height > data.length) {
    return null;
  }

  const matrix = [];
  for (let row = 0; row < height; row += 1) {
    const start = row * width;
    matrix.push(data.slice(start, start + width).map((value) => finiteNumber(value)));
  }
  return matrix;
}

function normalizeChannel(name, candidate, parentGrid) {
  const config = candidate && typeof candidate === "object" && !Array.isArray(candidate) ? candidate : {};
  const rawData = Array.isArray(candidate)
    ? candidate
    : config.values || config.data || config.grid || config.field || null;
  const width = config.width ?? config.nx ?? config.shape?.[1] ?? parentGrid?.width ?? parentGrid?.nx ?? parentGrid?.shape?.[1];
  const height = config.height ?? config.ny ?? config.shape?.[0] ?? parentGrid?.height ?? parentGrid?.ny ?? parentGrid?.shape?.[0];
  const matrix = matrixFromData(rawData, width, height);
  if (!matrix) {
    return null;
  }

  const values = matrix.flat().filter(Number.isFinite);
  if (values.length === 0) {
    return null;
  }

  const key = String(config.name || name || "field").toLowerCase();
  return {
    key,
    label: config.label || CHANNEL_LABELS[key] || toTitleCase(config.name || name || "field"),
    unit: config.unit || CHANNEL_UNITS[key] || "",
    matrix,
    min: Math.min(...values),
    max: Math.max(...values),
    xCoords: config.x_coords || parentGrid?.x_coords || parentGrid?.x || null,
    yCoords: config.y_coords || parentGrid?.y_coords || parentGrid?.y || null,
  };
}

function normalizeFieldGrid(fieldGrid) {
  if (!fieldGrid || typeof fieldGrid !== "object") {
    return null;
  }

  const channels = [];
  const seen = new Set();
  const pushChannel = (name, candidate) => {
    const channel = normalizeChannel(name, candidate, fieldGrid);
    if (!channel || seen.has(channel.key)) {
      return;
    }
    seen.add(channel.key);
    channels.push(channel);
  };

  if (Array.isArray(fieldGrid.channels)) {
    fieldGrid.channels.forEach((channel, index) => {
      pushChannel(channel?.name || channel?.key || `channel_${index + 1}`, channel);
    });
  } else if (fieldGrid.channels && typeof fieldGrid.channels === "object") {
    Object.entries(fieldGrid.channels).forEach(([name, candidate]) => {
      pushChannel(name, candidate);
    });
  }

  if (Array.isArray(fieldGrid.data) && Array.isArray(fieldGrid.channel_names)) {
    fieldGrid.channel_names.forEach((name, index) => {
      pushChannel(name, {
        data: fieldGrid.data[index],
        unit: fieldGrid.units?.[index] || fieldGrid.channel_units?.[index],
      });
    });
  } else if (Array.isArray(fieldGrid.data)) {
    pushChannel(fieldGrid.name || "field", fieldGrid);
  }

  [
    "temperature",
    "temperature_k",
    "temperature_perturbation",
    "temperature_perturbation_k",
    "displacement_magnitude",
    "displacement_magnitude_m",
    "magnitude_m",
  ].forEach((name) => {
    if (fieldGrid[name]) {
      pushChannel(name, fieldGrid[name]);
    }
  });

  return channels.length > 0 ? { channels } : null;
}

function colorForValue(value, min, max) {
  if (!Number.isFinite(value)) {
    return "rgba(15, 23, 42, 0.2)";
  }
  const span = Math.abs(max - min) < 1e-12 ? 1 : max - min;
  const t = clamp((value - min) / span, 0, 1);
  const stops = [
    [30, 41, 59],
    [14, 165, 233],
    [34, 197, 94],
    [245, 158, 11],
  ];
  const segment = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const localT = t * (stops.length - 1) - segment;
  const start = stops[segment];
  const end = stops[segment + 1];
  const rgb = start.map((component, index) => Math.round(component + (end[index] - component) * localT));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function coordAt(coords, index, count) {
  if (Array.isArray(coords) && Number.isFinite(Number(coords[index]))) {
    return Number(coords[index]);
  }
  return count <= 1 ? 0 : index / (count - 1);
}

function buildHeatmapSvg(channel) {
  const rows = channel.matrix.length;
  const cols = Math.max(...channel.matrix.map((row) => row.length));
  const width = 760;
  const height = 500;
  const plot = { x: 64, y: 38, width: 560, height: 372 };
  const cellWidth = plot.width / cols;
  const cellHeight = plot.height / rows;
  const unitSuffix = channel.unit ? ` ${channel.unit}` : "";
  const cells = [];

  channel.matrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const x = plot.x + colIndex * cellWidth;
      const y = plot.y + (rows - 1 - rowIndex) * cellHeight;
      const coordX = coordAt(channel.xCoords, colIndex, cols);
      const coordY = coordAt(channel.yCoords, rowIndex, rows);
      const valueLabel = Number.isFinite(value) ? `${formatNumber(value, 5)}${unitSuffix}` : "missing";
      cells.push(`
        <rect
          x="${x}" y="${y}"
          width="${Math.ceil(cellWidth) + 0.25}" height="${Math.ceil(cellHeight) + 0.25}"
          fill="${colorForValue(value, channel.min, channel.max)}">
          <title>x=${formatCoordinate(coordX)}, y=${formatCoordinate(coordY)}, value=${escapeHtml(valueLabel)}</title>
        </rect>
      `);
    });
  });

  const legendX = plot.x + plot.width + 34;
  const legendY = plot.y;
  const legendHeight = plot.height;
  const legendSteps = 34;
  const legendRects = Array.from({ length: legendSteps }, (_, index) => {
    const t = index / (legendSteps - 1);
    const value = channel.min + (channel.max - channel.min) * (1 - t);
    return `<rect x="${legendX}" y="${legendY + index * (legendHeight / legendSteps)}" width="18" height="${Math.ceil(legendHeight / legendSteps) + 0.5}" fill="${colorForValue(value, channel.min, channel.max)}" />`;
  }).join("");

  return `
    <svg class="heatmap-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(channel.label)} heatmap">
      <rect x="0" y="0" width="${width}" height="${height}" rx="24" fill="rgba(15, 23, 42, 0.24)" />
      <text x="${plot.x}" y="24" font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" font-weight="700" fill="${COLORS.label}">
        ${escapeHtml(channel.label)}
      </text>
      <g>
        <rect x="${plot.x}" y="${plot.y}" width="${plot.width}" height="${plot.height}" fill="rgba(15, 23, 42, 0.32)" stroke="${COLORS.tooltipStroke}" />
        ${cells.join("")}
      </g>
      <text x="${plot.x}" y="${plot.y + plot.height + 28}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">x · 0 to 1 m</text>
      <text x="${plot.x - 44}" y="${plot.y + 14}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">y</text>
      <text x="${plot.x - 28}" y="${plot.y + plot.height}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">0</text>
      <text x="${plot.x - 28}" y="${plot.y + 10}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">1</text>
      <g>
        ${legendRects}
        <rect x="${legendX}" y="${legendY}" width="18" height="${legendHeight}" fill="none" stroke="${COLORS.tooltipStroke}" />
        <text x="${legendX + 30}" y="${legendY + 10}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.label}">${formatNumber(channel.max, 4)}${unitSuffix}</text>
        <text x="${legendX + 30}" y="${legendY + legendHeight}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.label}">${formatNumber(channel.min, 4)}${unitSuffix}</text>
      </g>
      <text x="${plot.x}" y="${height - 24}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="12.5" fill="${COLORS.meta}">
        Hover cells to inspect coordinate and value.
      </text>
    </svg>
  `;
}

function renderUnavailableHeatmap(container) {
  container.innerHTML = `
    <div class="heatmap-empty">
      Spatial field grid is not returned for this model/request.
    </div>
  `;
}

export function renderFieldGridHeatmap(container, fieldGrid) {
  if (!container) {
    return;
  }

  if (fieldGrid === undefined) {
    container.innerHTML = "";
    return;
  }

  const parsed = normalizeFieldGrid(fieldGrid);
  if (!parsed) {
    renderUnavailableHeatmap(container);
    return;
  }

  const options = parsed.channels
    .map((channel, index) => `<option value="${index}">${escapeHtml(channel.label)}</option>`)
    .join("");

  container.innerHTML = `
    <div class="heatmap-controls">
      <label class="heatmap-control">
        <span>Channel</span>
        <select class="heatmap-channel-select">${options}</select>
      </label>
      <div class="heatmap-stats" aria-live="polite"></div>
    </div>
    <div class="heatmap-figure"></div>
  `;

  const select = container.querySelector(".heatmap-channel-select");
  const stats = container.querySelector(".heatmap-stats");
  const figure = container.querySelector(".heatmap-figure");

  const draw = (index) => {
    const channel = parsed.channels[index] || parsed.channels[0];
    const unitSuffix = channel.unit ? ` ${channel.unit}` : "";
    stats.textContent = `min ${formatNumber(channel.min, 5)}${unitSuffix} · max ${formatNumber(channel.max, 5)}${unitSuffix}`;
    figure.innerHTML = buildHeatmapSvg(channel);
  };

  select.addEventListener("change", () => {
    draw(Number(select.value));
  });

  draw(0);
}
