function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeComponent(value, maxValue) {
  if (maxValue <= 0) {
    return 0;
  }
  return clamp(value / maxValue, 0, 1);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

const COLORS = {
  domainFillFront: "#172033",
  domainFillBack: "rgba(51, 65, 85, 0.34)",
  domainStroke: "rgba(148, 163, 184, 0.88)",
  grid: "rgba(148, 163, 184, 0.22)",
  link: "rgba(226, 232, 240, 0.52)",
  sourceFill: "#f59e0b",
  sourceStroke: "#fde68a",
  probeFill: "#38bdf8",
  probeStroke: "#bae6fd",
  arrow: "#f8fafc",
  arrowGlow: "rgba(248, 250, 252, 0.18)",
  label: "#f8fafc",
  meta: "#94a3b8",
  shadow: "rgba(15, 23, 42, 0.62)",
};

function projectPoint(point, box) {
  const { width, height, margin, skewX, skewY } = box;
  const innerWidth = width - margin * 2 - skewX;
  const innerHeight = height - margin * 2 - skewY;
  return {
    x: margin + point.x * innerWidth + point.z * skewX,
    y: height - margin - point.y * innerHeight - point.z * skewY,
  };
}

function buildBoxGeometry(domain) {
  const width = 1000;
  const height = 650;
  const margin = 90;
  const skewX = domain.type === "rect_3d" ? 120 : 0;
  const skewY = domain.type === "rect_3d" ? 80 : 0;
  return { width, height, margin, skewX, skewY };
}

function normalizedDomainPoint(point, domain) {
  return {
    x: normalizeComponent(Number(point.x) || 0, Math.max(domain.size.lx, 1e-8)),
    y: normalizeComponent(Number(point.y) || 0, Math.max(domain.size.ly, 1e-8)),
    z:
      domain.type === "rect_3d"
        ? normalizeComponent(Number(point.z) || 0, Math.max(domain.size.lz, 1e-8))
        : 0,
  };
}

function buildGridLines(box, domain) {
  const lines = [];
  const steps = 5;
  for (let index = 1; index < steps; index += 1) {
    const t = index / steps;
    if (domain.type === "rect_3d") {
      const frontLeft = projectPoint({ x: 0, y: t, z: 0 }, box);
      const frontRight = projectPoint({ x: 1, y: t, z: 0 }, box);
      const depthLeft = projectPoint({ x: 0, y: t, z: 1 }, box);
      const bottomFront = projectPoint({ x: t, y: 0, z: 0 }, box);
      const topFront = projectPoint({ x: t, y: 1, z: 0 }, box);
      const topBack = projectPoint({ x: t, y: 1, z: 1 }, box);
      const frontDepth = projectPoint({ x: t, y: 0, z: 0 }, box);
      const backDepth = projectPoint({ x: t, y: 0, z: 1 }, box);

      lines.push(
        `<line x1="${frontLeft.x}" y1="${frontLeft.y}" x2="${frontRight.x}" y2="${frontRight.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
      lines.push(
        `<line x1="${frontLeft.x}" y1="${frontLeft.y}" x2="${depthLeft.x}" y2="${depthLeft.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
      lines.push(
        `<line x1="${bottomFront.x}" y1="${bottomFront.y}" x2="${topFront.x}" y2="${topFront.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
      lines.push(
        `<line x1="${frontDepth.x}" y1="${frontDepth.y}" x2="${backDepth.x}" y2="${backDepth.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
      lines.push(
        `<line x1="${topFront.x}" y1="${topFront.y}" x2="${topBack.x}" y2="${topBack.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
    } else {
      const verticalStart = projectPoint({ x: t, y: 0, z: 0 }, box);
      const verticalEnd = projectPoint({ x: t, y: 1, z: 0 }, box);
      const horizontalStart = projectPoint({ x: 0, y: t, z: 0 }, box);
      const horizontalEnd = projectPoint({ x: 1, y: t, z: 0 }, box);
      lines.push(
        `<line x1="${verticalStart.x}" y1="${verticalStart.y}" x2="${verticalEnd.x}" y2="${verticalEnd.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
      lines.push(
        `<line x1="${horizontalStart.x}" y1="${horizontalStart.y}" x2="${horizontalEnd.x}" y2="${horizontalEnd.y}" stroke="${COLORS.grid}" stroke-width="1.1" />`
      );
    }
  }
  return lines.join("");
}

function buildDomainMarkup(box, domain) {
  if (domain.type === "rect_3d") {
    const front = [
      projectPoint({ x: 0, y: 0, z: 0 }, box),
      projectPoint({ x: 1, y: 0, z: 0 }, box),
      projectPoint({ x: 1, y: 1, z: 0 }, box),
      projectPoint({ x: 0, y: 1, z: 0 }, box),
    ];
    const back = [
      projectPoint({ x: 0, y: 0, z: 1 }, box),
      projectPoint({ x: 1, y: 0, z: 1 }, box),
      projectPoint({ x: 1, y: 1, z: 1 }, box),
      projectPoint({ x: 0, y: 1, z: 1 }, box),
    ];
    const face = (points) => points.map((point) => `${point.x},${point.y}`).join(" ");
    const edges = front
      .map((point, index) => {
        const backPoint = back[index];
        return `<line x1="${point.x}" y1="${point.y}" x2="${backPoint.x}" y2="${backPoint.y}" stroke="${COLORS.domainStroke}" stroke-width="2.2" />`;
      })
      .join("");

    return `
      <polygon points="${face(back)}" fill="${COLORS.domainFillBack}" stroke="${COLORS.domainStroke}" stroke-width="2.5" />
      <polygon points="${face(front)}" fill="${COLORS.domainFillFront}" stroke="${COLORS.domainStroke}" stroke-width="2.8" />
      ${edges}
    `;
  }

  const bottomLeft = projectPoint({ x: 0, y: 0, z: 0 }, box);
  const bottomRight = projectPoint({ x: 1, y: 0, z: 0 }, box);
  const topRight = projectPoint({ x: 1, y: 1, z: 0 }, box);
  const topLeft = projectPoint({ x: 0, y: 1, z: 0 }, box);
  return `
    <polygon
      points="${bottomLeft.x},${bottomLeft.y} ${bottomRight.x},${bottomRight.y} ${topRight.x},${topRight.y} ${topLeft.x},${topLeft.y}"
      fill="${COLORS.domainFillFront}" stroke="${COLORS.domainStroke}" stroke-width="2.8" />
  `;
}

function buildPointMarkup({ point, shadowPoint, label, fill, stroke, depthLabel, labelOffsetX, labelOffsetY }) {
  return `
    <line x1="${shadowPoint.x}" y1="${shadowPoint.y}" x2="${point.x}" y2="${point.y}" stroke="${COLORS.link}" stroke-width="1.8" stroke-dasharray="5 5" />
    <ellipse cx="${shadowPoint.x}" cy="${shadowPoint.y + 7}" rx="18" ry="7" fill="${COLORS.shadow}" opacity="0.42" />
    <circle cx="${point.x}" cy="${point.y}" r="16" fill="${fill}" stroke="${stroke}" stroke-width="4" />
    <text x="${point.x + labelOffsetX}" y="${point.y + labelOffsetY}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" fill="${COLORS.label}">${label}</text>
    <text x="${point.x + labelOffsetX}" y="${point.y + labelOffsetY + 18}" font-family="JetBrains Mono, monospace" font-size="12" fill="${COLORS.meta}">${depthLabel}</text>
  `;
}

function buildArrowMarkup(box, payload, response, probePoint) {
  if (!response?.prediction?.direction_vector) {
    return "";
  }

  const [vx = 0, vy = 0, vz = 0] = response.prediction.direction_vector;
  const directionMagnitude = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
  const scaled = {
    x: vx / directionMagnitude,
    y: vy / directionMagnitude,
    z: vz / directionMagnitude,
  };

  const arrowLength = payload.domain.type === "rect_3d" ? 170 : 220;
  const projectedEnd = {
    x: clamp(probePoint.x + scaled.x * arrowLength + scaled.z * box.skewX * 0.75, box.margin - 10, box.width - box.margin + box.skewX + 10),
    y: clamp(probePoint.y - scaled.y * arrowLength - scaled.z * box.skewY * 0.75, box.margin - box.skewY - 10, box.height - box.margin + 10),
  };

  return `
    <line
      x1="${probePoint.x}" y1="${probePoint.y}"
      x2="${projectedEnd.x}" y2="${projectedEnd.y}"
      stroke="${COLORS.arrowGlow}" stroke-width="11" stroke-linecap="round" />
    <line
      class="viz-arrow"
      x1="${probePoint.x}" y1="${probePoint.y}"
      x2="${projectedEnd.x}" y2="${projectedEnd.y}"
      stroke="${COLORS.arrow}" stroke-width="5.5" stroke-linecap="round" fill="none"
      marker-end="url(#arrowhead-3d)" />
    <text x="${projectedEnd.x - 14}" y="${projectedEnd.y - 18}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="14" fill="${COLORS.label}">
      Predicted direction
    </text>
  `;
}

function buildAxisLabels(box, domain) {
  const xBase = projectPoint({ x: 1, y: 0, z: 0 }, box);
  const yBase = projectPoint({ x: 0, y: 1, z: 0 }, box);
  const zBase = projectPoint({ x: 0, y: 0, z: 1 }, box);
  const xLabel = `<text x="${xBase.x + 10}" y="${xBase.y + 22}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">x · ${formatNumber(domain.size.lx)}</text>`;
  const yLabel = `<text x="${yBase.x - 36}" y="${yBase.y - 10}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">y · ${formatNumber(domain.size.ly)}</text>`;
  if (domain.type !== "rect_3d") {
    return `${xLabel}${yLabel}`;
  }
  const zLabel = `<text x="${zBase.x - 24}" y="${zBase.y - 12}" font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">z · ${formatNumber(domain.size.lz)}</text>`;
  return `${xLabel}${yLabel}${zLabel}`;
}

export function renderDomain(svg, payload, response, modelLabel) {
  if (!svg || !payload) {
    return;
  }

  const box = buildBoxGeometry(payload.domain);
  const normalizedSource = normalizedDomainPoint(payload.source, payload.domain);
  const normalizedProbe = normalizedDomainPoint(payload.probe, payload.domain);
  const sourcePoint = projectPoint(normalizedSource, box);
  const probePoint = projectPoint(normalizedProbe, box);
  const sourceShadow = projectPoint({ ...normalizedSource, z: 0 }, box);
  const probeShadow = projectPoint({ ...normalizedProbe, z: 0 }, box);
  const is3d = payload.domain.type === "rect_3d";
  const domainSummary = is3d
    ? `Domain rect_3d | ${formatNumber(payload.domain.size.lx)} x ${formatNumber(payload.domain.size.ly)} x ${formatNumber(payload.domain.size.lz)} | ${payload.domain.resolution.nx} x ${payload.domain.resolution.ny} x ${payload.domain.resolution.nz}`
    : `Domain rect_2d | ${formatNumber(payload.domain.size.lx)} x ${formatNumber(payload.domain.size.ly)} | ${payload.domain.resolution.nx} x ${payload.domain.resolution.ny}`;

  svg.innerHTML = `
    <defs>
      <marker id="arrowhead-3d" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${COLORS.arrow}"></path>
      </marker>
    </defs>
    ${buildDomainMarkup(box, payload.domain)}
    ${buildGridLines(box, payload.domain)}
    <line x1="${sourcePoint.x}" y1="${sourcePoint.y}" x2="${probePoint.x}" y2="${probePoint.y}" stroke="${COLORS.link}" stroke-width="2.1" stroke-dasharray="7 7" />
    ${buildPointMarkup({
      point: sourcePoint,
      shadowPoint: sourceShadow,
      label: "Source",
      fill: COLORS.sourceFill,
      stroke: COLORS.sourceStroke,
      depthLabel: `z=${formatNumber(payload.source.z || 0)}`,
      labelOffsetX: 22,
      labelOffsetY: -16,
    })}
    ${buildPointMarkup({
      point: probePoint,
      shadowPoint: probeShadow,
      label: "Probe",
      fill: COLORS.probeFill,
      stroke: COLORS.probeStroke,
      depthLabel: `z=${formatNumber(payload.probe.z || 0)}`,
      labelOffsetX: 22,
      labelOffsetY: -16,
    })}
    ${buildArrowMarkup(box, payload, response, probePoint)}
    <text x="${box.margin}" y="52" font-family="Avenir Next, Segoe UI, sans-serif" font-size="16" font-weight="600" fill="${COLORS.label}">
      ${modelLabel || "No model selected"}
    </text>
    <text x="${box.margin}" y="76" font-family="Avenir Next, Segoe UI, sans-serif" font-size="13" fill="${COLORS.meta}">
      ${domainSummary}
    </text>
    <text x="${box.margin}" y="${box.height - 26}" font-family="Avenir Next, Segoe UI, sans-serif" font-size="12.5" fill="${COLORS.meta}">
      ${is3d ? "Projected 3D preview using the current domain extents and source/probe depth." : "Planar preview for 2D-compatible routes."}
    </text>
    ${buildAxisLabels(box, payload.domain)}
  `;
}
