function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

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
    gridLines.push(`<line x1="${x}" y1="${margin}" x2="${x}" y2="${height - margin}" class="viz-grid" />`);
    gridLines.push(`<line x1="${margin}" y1="${y}" x2="${width - margin}" y2="${y}" class="viz-grid" />`);
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
      <line class="viz-arrow" x1="${probePoint.x}" y1="${probePoint.y}" x2="${arrowEnd.x}" y2="${arrowEnd.y}" marker-end="url(#arrowhead)" />
      <text x="${arrowEnd.x - 12}" y="${arrowEnd.y - 18}" class="viz-label viz-label--arrow">Predicted direction</text>
    `;
  }

  svg.innerHTML = `
    <defs>
      <linearGradient id="domainFill" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="rgba(117,244,208,0.18)" />
        <stop offset="100%" stop-color="rgba(91,123,255,0.08)" />
      </linearGradient>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#5ed9ff"></path>
      </marker>
    </defs>
    <style>
      .viz-domain { fill: rgba(10, 24, 40, 0.62); stroke: rgba(117, 244, 208, 0.28); stroke-width: 2; }
      .viz-grid { stroke: rgba(149, 167, 191, 0.12); stroke-width: 1; }
      .viz-link { stroke: rgba(248, 187, 103, 0.45); stroke-width: 2; stroke-dasharray: 8 8; }
      .viz-source { fill: #f8bb67; stroke: rgba(248, 187, 103, 0.36); stroke-width: 8; }
      .viz-probe { fill: #75f4d0; stroke: rgba(117, 244, 208, 0.28); stroke-width: 8; }
      .viz-arrow { stroke: #5ed9ff; stroke-width: 5; stroke-linecap: round; fill: none; }
      .viz-label { font-family: "IBM Plex Sans", sans-serif; font-size: 15px; fill: #dce7f8; }
      .viz-label--meta { fill: #9aacbf; }
      .viz-axis { font-family: "JetBrains Mono", monospace; font-size: 13px; fill: #8fa5c2; }
    </style>
    <rect x="${margin}" y="${margin}" width="${width - margin * 2}" height="${height - margin * 2}" rx="32" class="viz-domain" />
    ${gridLines.join("")}
    <line x1="${sourcePoint.x}" y1="${sourcePoint.y}" x2="${probePoint.x}" y2="${probePoint.y}" class="viz-link" />
    <circle cx="${sourcePoint.x}" cy="${sourcePoint.y}" r="10" class="viz-source" />
    <circle cx="${probePoint.x}" cy="${probePoint.y}" r="10" class="viz-probe" />
    <text x="${sourcePoint.x + 16}" y="${sourcePoint.y - 14}" class="viz-label">Source</text>
    <text x="${probePoint.x + 16}" y="${probePoint.y - 14}" class="viz-label">Probe</text>
    <text x="${margin}" y="52" class="viz-label">${modelLabel || "No model selected"}</text>
    <text x="${margin}" y="76" class="viz-label viz-label--meta">Domain ${domain.type} | ${domain.size.lx} x ${domain.size.ly} | ${domain.resolution.nx} x ${domain.resolution.ny}</text>
    <text x="${margin}" y="${height - 30}" class="viz-axis">x</text>
    <text x="${margin - 28}" y="${margin}" class="viz-axis">y</text>
    ${arrowMarkup}
  `;

  const arrow = svg.querySelector(".viz-arrow");
  if (arrow) {
    requestAnimationFrame(() => arrow.classList.add("is-active"));
  }
}
