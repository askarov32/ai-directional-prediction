function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

const COLORS = {
  domainFill: "#0f172a",
  domainStroke: "rgba(148, 163, 184, 0.85)",
  grid: "rgba(148, 163, 184, 0.35)",
  link: "rgba(226, 232, 240, 0.7)",
  sourceFill: "#f59e0b",
  sourceStroke: "#fde68a",
  probeFill: "#38bdf8",
  probeStroke: "#bae6fd",
  arrow: "#fde68a",
  label: "#f8fafc",
  meta: "#94a3b8",
};

export function renderDomain(svg, payload, response, modelLabel) {
  if (!svg || !payload) {
    return;
  }

  const width = 1000;
  const height = 650;
  const margin = 90;
  const domain = payload.domain;
  const maxX = Math.max(domain.size.lx, 0.01);
  const maxY = Math.max(domain.size.ly, 0.01);

  const project = (x, y) => ({
    x: margin + (x / maxX) * (width - margin * 2),
    y: height - margin - (y / maxY) * (height - margin * 2),
  });

  const sourcePoint = project(payload.source.x, payload.source.y);
  const probePoint = project(payload.probe.x, payload.probe.y);

  const gridLines = [];
  for (let index = 0; index <= 8; index += 1) {
    const x = margin + ((width - margin * 2) / 8) * index;
    const y = margin + ((height - margin * 2) / 8) * index;
    gridLines.push(
      `<line x1="${x}" y1="${margin}" x2="${x}" y2="${height - margin}" stroke="${COLORS.grid}" stroke-width="1.2" />`
    );
    gridLines.push(
      `<line x1="${margin}" y1="${y}" x2="${width - margin}" y2="${y}" stroke="${COLORS.grid}" stroke-width="1.2" />`
    );
  }

  let arrowMarkup = "";
  if (response?.prediction?.direction_vector) {
    const [vx, vy] = response.prediction.direction_vector;
    const arrowLength = 220;
    const arrowEnd = {
      x: clamp(probePoint.x + vx * arrowLength, margin, width - margin),
      y: clamp(probePoint.y - vy * arrowLength, margin, height - margin),
    };

    arrowMarkup = `
      <line class="viz-arrow"
        x1="${probePoint.x}" y1="${probePoint.y}"
        x2="${arrowEnd.x}" y2="${arrowEnd.y}"
        stroke="${COLORS.arrow}" stroke-width="6" stroke-linecap="round" fill="none"
        marker-end="url(#arrowhead)" />
      <text x="${arrowEnd.x - 12}" y="${arrowEnd.y - 18}"
        font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" fill="${COLORS.label}">Predicted direction</text>
    `;
  }

  svg.innerHTML = `
    <defs>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${COLORS.arrow}"></path>
      </marker>
    </defs>
    <rect class="viz-domain"
      x="${margin}" y="${margin}"
      width="${width - margin * 2}" height="${height - margin * 2}" rx="32"
      fill="${COLORS.domainFill}" stroke="${COLORS.domainStroke}" stroke-width="3" />
    ${gridLines.join("")}
    <line x1="${sourcePoint.x}" y1="${sourcePoint.y}" x2="${probePoint.x}" y2="${probePoint.y}"
      stroke="${COLORS.link}" stroke-width="2.5" stroke-dasharray="7 7" />
    <circle class="viz-source" cx="${sourcePoint.x}" cy="${sourcePoint.y}" r="18"
      fill="${COLORS.sourceFill}" stroke="${COLORS.sourceStroke}" stroke-width="4" />
    <circle class="viz-probe" cx="${probePoint.x}" cy="${probePoint.y}" r="18"
      fill="${COLORS.probeFill}" stroke="${COLORS.probeStroke}" stroke-width="4" />
    <text x="${sourcePoint.x + 22}" y="${sourcePoint.y - 18}"
      font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" fill="${COLORS.label}">Source</text>
    <text x="${probePoint.x + 22}" y="${probePoint.y - 18}"
      font-family="Avenir Next, Segoe UI, sans-serif" font-size="15" fill="${COLORS.label}">Probe</text>
    <text x="${margin}" y="52"
      font-family="Avenir Next, Segoe UI, sans-serif" font-size="16" font-weight="600" fill="${COLORS.label}">${modelLabel || "No model selected"}</text>
    <text x="${margin}" y="76"
      font-family="Avenir Next, Segoe UI, sans-serif" font-size="13" fill="${COLORS.meta}">Domain ${domain.type} | ${domain.size.lx} x ${domain.size.ly} | ${domain.resolution.nx} x ${domain.resolution.ny}</text>
    <text x="${margin}" y="${height - 30}"
      font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">x</text>
    <text x="${margin - 28}" y="${margin}"
      font-family="JetBrains Mono, monospace" font-size="13" fill="${COLORS.meta}">y</text>
    ${arrowMarkup}
  `;
}
