const PROPERTY_META = [
  ["rho", "Density"],
  ["porosity_total", "Total porosity"],
  ["porosity_effective", "Effective porosity"],
  ["vp", "P-wave velocity"],
  ["vs", "S-wave velocity"],
  ["thermal_conductivity", "Thermal conductivity"],
  ["heat_capacity", "Heat capacity"],
  ["thermal_expansion", "Thermal expansion"],
];

function prettyNumber(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (Math.abs(value) >= 1000) {
    return value.toFixed(1);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(3);
  }
  if (Math.abs(value) >= 0.01) {
    return value.toFixed(4);
  }
  return value.toExponential(2);
}

const NOT_RETURNED = "Not returned";
const PROTOTYPE_NOTE = "Prototype prediction; not a field-validated thermoelastic simulation.";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function finiteValue(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
}

function sourceLabel(value) {
  if (!value) {
    return "Unavailable";
  }
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatFixed(value, decimals, unit = "") {
  const numeric = finiteValue(value);
  if (!Number.isFinite(numeric)) {
    return NOT_RETURNED;
  }
  return `${numeric.toFixed(decimals)}${unit ? ` ${unit}` : ""}`;
}

function formatDisplacement(value) {
  const numeric = finiteValue(value);
  if (!Number.isFinite(numeric)) {
    return NOT_RETURNED;
  }
  if (numeric === 0 || Math.abs(numeric) < 1e-3) {
    return `${numeric.toExponential(3)} m`;
  }
  return `${numeric.toFixed(6)} m`;
}

function formatVector(unitDir) {
  const x = finiteValue(unitDir?.x);
  const y = finiteValue(unitDir?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return NOT_RETURNED;
  }
  return `[${x.toFixed(3)}, ${y.toFixed(3)}]`;
}

function setText(node, value) {
  if (node) {
    node.textContent = value;
  }
}

function setHidden(node, hidden) {
  if (node) {
    node.classList.toggle("is-hidden", hidden);
  }
}

function normalizeTextList(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item || "").trim())
      .filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) {
    return [value.trim()];
  }
  return [];
}

function renderList(node, items, emptyText) {
  if (!node) {
    return;
  }
  const values = items.length > 0 ? items : [emptyText];
  node.innerHTML = values.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function uniqueList(items) {
  return Array.from(new Set(items));
}

function renderDisplacementVector(container, uValue, vValue, magnitudeValue) {
  if (!container) {
    return;
  }

  const u = finiteValue(uValue);
  const v = finiteValue(vValue);
  const magnitude = finiteValue(magnitudeValue);
  if (!Number.isFinite(u) || !Number.isFinite(v)) {
    container.innerHTML = `<div class="displacement-vector-empty">${NOT_RETURNED}</div>`;
    return;
  }

  const scaleBase = Math.max(Math.abs(u), Math.abs(v), Number.isFinite(magnitude) ? magnitude : 0, 1e-12);
  const arrowLength = 42;
  const endX = 90 + (u / scaleBase) * arrowLength;
  const endY = 58 - (v / scaleBase) * arrowLength;
  const uPercent = Math.min(100, (Math.abs(u) / scaleBase) * 100);
  const vPercent = Math.min(100, (Math.abs(v) / scaleBase) * 100);

  container.innerHTML = `
    <div class="displacement-vector-layout">
      <svg class="displacement-vector-svg" viewBox="0 0 180 116" role="img" aria-label="Displacement vector preview">
        <defs>
          <marker id="disp-arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#e2e8f0"></path>
          </marker>
        </defs>
        <line x1="32" y1="58" x2="148" y2="58" stroke="rgba(148, 163, 184, 0.35)" stroke-width="1" />
        <line x1="90" y1="100" x2="90" y2="16" stroke="rgba(148, 163, 184, 0.35)" stroke-width="1" />
        <circle cx="90" cy="58" r="4" fill="#94a3b8" />
        <line x1="90" y1="58" x2="${endX}" y2="${endY}" stroke="#e2e8f0" stroke-width="4" stroke-linecap="round" marker-end="url(#disp-arrowhead)" />
      </svg>
      <div class="component-bars">
        <div>
          <span>u</span>
          <i style="--bar-width: ${uPercent}%"></i>
          <strong>${escapeHtml(formatDisplacement(u))}</strong>
        </div>
        <div>
          <span>v</span>
          <i style="--bar-width: ${vPercent}%"></i>
          <strong>${escapeHtml(formatDisplacement(v))}</strong>
        </div>
      </div>
    </div>
  `;
}

export function createUI() {
  const refs = {
    form: document.querySelector("#prediction-form"),
    mediumSelect: document.querySelector("#medium-select"),
    modelSelect: document.querySelector("#model-select"),
    modelHint: document.querySelector("#model-capability-hint"),
    mediumProperties: document.querySelector("#medium-properties"),
    predictButton: document.querySelector("#predict-button"),
    demoButton: document.querySelector("#demo-button"),
    resetButton: document.querySelector("#reset-button"),
    resultCard: document.querySelector("#result-card"),
    resultEmpty: document.querySelector("#result-empty"),
    resultContent: document.querySelector("#result-content"),
    errorBanner: document.querySelector("#error-banner"),
    thermalNote: document.querySelector("#thermal-note"),
    thermalTemperature: document.querySelector("#thermal-temperature"),
    thermalTemperatureSource: document.querySelector("#thermal-temperature-source"),
    thermalPerturbation: document.querySelector("#thermal-perturbation"),
    thermalPerturbationSource: document.querySelector("#thermal-perturbation-source"),
    displacementSource: document.querySelector("#displacement-source"),
    displacementU: document.querySelector("#displacement-u"),
    displacementV: document.querySelector("#displacement-v"),
    displacementMagnitude: document.querySelector("#displacement-magnitude"),
    displacementVector: document.querySelector("#displacement-vector"),
    directionDistance: document.querySelector("#direction-distance"),
    directionAzimuth: document.querySelector("#direction-azimuth"),
    directionUnit: document.querySelector("#direction-unit"),
    directionScore: document.querySelector("#direction-score"),
    temporalSource: document.querySelector("#temporal-source"),
    travelTimeS: document.querySelector("#travel-time-s"),
    travelTimeMs: document.querySelector("#travel-time-ms"),
    metadataModel: document.querySelector("#metadata-model"),
    metadataVersion: document.querySelector("#metadata-version"),
    metadataLatency: document.querySelector("#metadata-latency"),
    metadataRequestId: document.querySelector("#metadata-request-id"),
    metadataFallback: document.querySelector("#metadata-fallback"),
    metadataFallbackReason: document.querySelector("#metadata-fallback-reason"),
    diagnosticsStatusBadge: document.querySelector("#diagnostics-status-badge"),
    diagnosticsFallbackBadge: document.querySelector("#diagnostics-fallback-badge"),
    diagnosticsWarningBadge: document.querySelector("#diagnostics-warning-badge"),
    diagnosticsRequestId: document.querySelector("#diagnostics-request-id"),
    diagnosticsSchemaVersion: document.querySelector("#diagnostics-schema-version"),
    diagnosticsFallbackReason: document.querySelector("#diagnostics-fallback-reason"),
    diagnosticsWarnings: document.querySelector("#diagnostics-warnings"),
    diagnosticsNotes: document.querySelector("#diagnostics-notes"),
    latencyBadge: document.querySelector("#latency-badge"),
    modelBadge: document.querySelector("#model-badge"),
    requestJson: document.querySelector("#request-json"),
    responseJson: document.querySelector("#response-json"),
    copyJsonButton: document.querySelector("#copy-json-button"),
    domainSvg: document.querySelector("#domain-svg"),
    fieldGridHeatmap: document.querySelector("#field-grid-heatmap"),
    fieldErrors: Array.from(document.querySelectorAll("[data-error-for]")),
  };

  function setSelectOptions(select, options, selectedValue) {
    select.innerHTML = options
      .map(
        (option) =>
          `<option value="${option.value}" ${option.value === selectedValue ? "selected" : ""}>${option.label}</option>`
      )
      .join("");
  }

  function renderMediaOptions(media, selectedMediumId) {
    setSelectOptions(
      refs.mediumSelect,
      media.map((medium) => ({
        value: medium.id,
        label: medium.name,
      })),
      selectedMediumId
    );
  }

  function renderModelOptions(models, selectedModel) {
    setSelectOptions(
      refs.modelSelect,
      models.map((model) => ({
        value: model.id,
        label: model.name,
      })),
      selectedModel
    );
    refs.modelHint.textContent = "Requests use the normalized 2D prediction contract.";
  }

  function renderMediumDetails(medium) {
    if (!medium) {
      refs.mediumProperties.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Property</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Medium</td>
              <td>No preset selected</td>
            </tr>
          </tbody>
        </table>
      `;
      return;
    }

    const rows = PROPERTY_META.map(([key, label]) => {
      const value = medium.properties[key];
      return `
        <tr>
          <td>${label}</td>
          <td>${prettyNumber(Number(value))}</td>
        </tr>
      `;
    }).join("");

    refs.mediumProperties.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Property</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Category</td>
            <td>${medium.category}</td>
          </tr>
          ${rows}
        </tbody>
      </table>
    `;

  }

  function renderValidation(errors) {
    refs.fieldErrors.forEach((element) => {
      element.textContent = errors[element.dataset.errorFor] || "";
    });
  }

  function setLoading(loading) {
    refs.predictButton.disabled = loading;
    refs.predictButton.classList.toggle("is-loading", loading);
  }

  function renderIdle(selectedModelLabel) {
    refs.resultCard.classList.remove("is-live");
    refs.resultEmpty.classList.remove("is-hidden");
    refs.resultContent.classList.add("is-hidden");
    refs.errorBanner.classList.add("is-hidden");
    refs.modelBadge.textContent = selectedModelLabel || "Awaiting prediction";
    refs.modelBadge.className = "model-pill model-pill--idle";
    refs.modelBadge.removeAttribute("data-model");
    refs.latencyBadge.textContent = "Latency --";
  }

  function _safeNum(value, fallback = NaN) {
    return Number.isFinite(Number(value)) ? Number(value) : fallback;
  }

  function renderDiagnostics(response, modelMeta = {}, diagnostics = {}) {
    const warnings = normalizeTextList(diagnostics.warnings);
    const rawNotes = normalizeTextList(diagnostics.notes);
    const notes = uniqueList([...rawNotes, PROTOTYPE_NOTE]);
    const fallbackUsed = Boolean(modelMeta.fallback_used);
    const fallbackReason = modelMeta.fallback_reason || diagnostics.fallback_reason || "";
    const schemaVersion = response.schema_version || NOT_RETURNED;
    const requestId = response.request_id || NOT_RETURNED;
    const hasWarnings = warnings.length > 0;

    setText(
      refs.diagnosticsStatusBadge,
      fallbackUsed ? "Fallback response" : hasWarnings ? "Warning response" : "Normal response"
    );
    refs.diagnosticsStatusBadge.className = fallbackUsed
      ? "diagnostics-badge diagnostics-badge--fallback"
      : hasWarnings
        ? "diagnostics-badge diagnostics-badge--warning"
        : "diagnostics-badge diagnostics-badge--normal";

    setText(refs.diagnosticsFallbackBadge, "Fallback response");
    setHidden(refs.diagnosticsFallbackBadge, !fallbackUsed);
    setText(refs.diagnosticsWarningBadge, hasWarnings ? `${warnings.length} warning${warnings.length === 1 ? "" : "s"}` : "Warnings");
    setHidden(refs.diagnosticsWarningBadge, !hasWarnings);
    setText(refs.diagnosticsRequestId, requestId);
    setText(refs.diagnosticsSchemaVersion, schemaVersion);
    setText(refs.diagnosticsFallbackReason, fallbackUsed ? `Fallback reason: ${fallbackReason || NOT_RETURNED}` : "");
    setHidden(refs.diagnosticsFallbackReason, !fallbackUsed);
    renderList(refs.diagnosticsWarnings, warnings, "No warnings returned.");
    renderList(refs.diagnosticsNotes, notes, PROTOTYPE_NOTE);
  }

  function renderResultV2(response, modelLabel) {
    refs.resultCard.classList.add("is-live");
    refs.resultEmpty.classList.add("is-hidden");
    refs.resultContent.classList.remove("is-hidden");
    refs.errorBanner.classList.add("is-hidden");

    const geom = response.geometry || {};
    const pred = response.prediction || {};
    const therm = pred.thermal || {};
    const disp = pred.displacement || {};
    const dir = pred.directional_response || {};
    const temporal = pred.temporal_response || {};
    const modelMeta = response.model || {};
    const diag = response.diagnostics || {};

    const temperature = therm.temperature_k || {};
    const theta = therm.temperature_perturbation_k || {};
    const temperatureK = _safeNum(temperature.value);
    const thetaK = _safeNum(theta.value);
    setText(refs.thermalTemperature, formatFixed(temperatureK, 3, "K"));
    setText(refs.thermalTemperatureSource, sourceLabel(temperature.source));
    setText(refs.thermalPerturbation, formatFixed(thetaK, 3, "K"));
    setText(refs.thermalPerturbationSource, sourceLabel(theta.source));
    setText(
      refs.thermalNote,
      Number.isFinite(temperatureK) || Number.isFinite(thetaK)
        ? "Kelvin-scale probe response"
        : "Thermal values not returned"
    );

    const components = disp.components_m || {};
    const u = _safeNum(components.u);
    const v = _safeNum(components.v);
    const magnitudeM = _safeNum(disp.magnitude_m);
    setText(refs.displacementU, formatDisplacement(u));
    setText(refs.displacementV, formatDisplacement(v));
    setText(refs.displacementMagnitude, formatDisplacement(magnitudeM));
    setText(refs.displacementSource, sourceLabel(disp.components_source || disp.magnitude_source));
    renderDisplacementVector(refs.displacementVector, u, v, magnitudeM);

    const unitDir = geom.unit_direction || {};
    const distanceM = _safeNum(dir.distance_m, _safeNum(geom.distance_m));
    const azimuthDeg = _safeNum(dir.azimuth_deg, _safeNum(geom.azimuth_deg));
    const score = _safeNum(dir.response_magnitude_score);
    setText(refs.directionDistance, formatFixed(distanceM, 3, "m"));
    setText(refs.directionAzimuth, formatFixed(azimuthDeg, 1, "deg"));
    setText(refs.directionUnit, formatVector(unitDir));
    setText(refs.directionScore, Number.isFinite(score) ? score.toFixed(3) : NOT_RETURNED);

    const travelTimeS = _safeNum(temporal.travel_time_s);
    const travelTimeMs = Number.isFinite(travelTimeS) ? travelTimeS * 1000.0 : NaN;
    setText(refs.travelTimeS, formatFixed(travelTimeS, 6, "s"));
    setText(refs.travelTimeMs, formatFixed(travelTimeMs, 3, "ms"));
    setText(refs.temporalSource, sourceLabel(temporal.source));

    setText(refs.metadataModel, modelMeta.name || modelLabel || NOT_RETURNED);
    setText(refs.metadataVersion, modelMeta.version || NOT_RETURNED);
    const latencyMs = _safeNum(modelMeta.inference_time_ms);
    setText(refs.metadataLatency, formatFixed(latencyMs, 1, "ms"));
    setText(refs.metadataRequestId, response.request_id || NOT_RETURNED);
    setText(refs.metadataFallback, modelMeta.fallback_used ? "Fallback used" : "No fallback");
    setText(
      refs.metadataFallbackReason,
      modelMeta.fallback_used ? `Reason: ${modelMeta.fallback_reason || "Not returned"}` : ""
    );
    setHidden(refs.metadataFallbackReason, !modelMeta.fallback_used);

    refs.latencyBadge.textContent = Number.isFinite(latencyMs)
      ? `Latency ${latencyMs.toFixed(1)} ms`
      : "Latency —";

    const fallback = modelMeta.fallback_used
      ? ` · fallback: ${modelMeta.fallback_reason || "yes"}`
      : "";
    refs.modelBadge.textContent =
      (modelLabel || modelMeta.name || "model") + fallback;
    refs.modelBadge.className = modelMeta.fallback_used
      ? "model-pill model-pill--fallback"
      : "model-pill";
    refs.modelBadge.dataset.model = modelMeta.name || "";
    if (Array.isArray(diag.warnings) && diag.warnings.length > 0 && !modelMeta.fallback_used) {
      setText(refs.metadataFallback, `Warnings: ${diag.warnings.length}`);
    }
    renderDiagnostics(response, modelMeta, diag);
  }

  function renderResult(response, modelLabel) {
    if (!response) {
      renderIdle(modelLabel);
      return;
    }
    if (response.schema_version === "2.0") {
      renderResultV2(response, modelLabel);
      return;
    }

    // Compatibility path for older backend responses during local demos.
    refs.resultCard.classList.add("is-live");
    refs.resultEmpty.classList.add("is-hidden");
    refs.resultContent.classList.remove("is-hidden");
    refs.errorBanner.classList.add("is-hidden");

    const planarDirection = (response.prediction.direction_vector || []).slice(0, 2);
    const legacyTravelTimeMs = _safeNum(response.prediction.travel_time_ms);
    const legacyDisplacement = _safeNum(response.field_summary.max_displacement);
    const legacyTheta = _safeNum(response.field_summary.max_temperature_perturbation);
    const legacyMagnitude = _safeNum(response.prediction.magnitude);

    setText(refs.thermalTemperature, NOT_RETURNED);
    setText(refs.thermalTemperatureSource, "Unavailable");
    setText(refs.thermalPerturbation, formatFixed(legacyTheta, 3, "K"));
    setText(refs.thermalPerturbationSource, Number.isFinite(legacyTheta) ? "Legacy field summary" : "Unavailable");
    setText(refs.thermalNote, "Legacy response mapped for display");

    setText(refs.displacementU, NOT_RETURNED);
    setText(refs.displacementV, NOT_RETURNED);
    setText(refs.displacementMagnitude, formatDisplacement(legacyDisplacement));
    setText(refs.displacementSource, Number.isFinite(legacyDisplacement) ? "Legacy field summary" : "Unavailable");
    renderDisplacementVector(refs.displacementVector, NaN, NaN, legacyDisplacement);

    setText(refs.directionDistance, NOT_RETURNED);
    setText(refs.directionAzimuth, formatFixed(response.prediction.azimuth_deg, 1, "deg"));
    setText(
      refs.directionUnit,
      planarDirection.length >= 2
        ? `[${planarDirection.map((item) => Number(item).toFixed(3)).join(", ")}]`
        : NOT_RETURNED
    );
    setText(refs.directionScore, Number.isFinite(legacyMagnitude) ? legacyMagnitude.toFixed(3) : NOT_RETURNED);

    setText(refs.travelTimeS, Number.isFinite(legacyTravelTimeMs) ? `${(legacyTravelTimeMs / 1000).toFixed(6)} s` : NOT_RETURNED);
    setText(refs.travelTimeMs, formatFixed(legacyTravelTimeMs, 3, "ms"));
    setText(refs.temporalSource, Number.isFinite(legacyTravelTimeMs) ? "Legacy prediction" : "Unavailable");

    setText(refs.metadataModel, response.model || modelLabel || NOT_RETURNED);
    setText(refs.metadataVersion, response.meta.model_version || NOT_RETURNED);
    setText(refs.metadataLatency, formatFixed(response.meta.latency_ms, 1, "ms"));
    setText(refs.metadataRequestId, response.meta.request_id || NOT_RETURNED);
    setText(refs.metadataFallback, "Legacy response");
    setText(refs.metadataFallbackReason, "");
    setHidden(refs.metadataFallbackReason, true);

    refs.latencyBadge.textContent = Number.isFinite(_safeNum(response.meta.latency_ms))
      ? `Latency ${Number(response.meta.latency_ms).toFixed(1)} ms`
      : "Latency —";
    refs.modelBadge.textContent = modelLabel || response.model;
    refs.modelBadge.className = "model-pill";
    refs.modelBadge.dataset.model = response.model;
    renderDiagnostics(
      {
        schema_version: response.schema_version || "1.0",
        request_id: response.meta?.request_id,
      },
      { fallback_used: false },
      { notes: [PROTOTYPE_NOTE], warnings: [] }
    );
  }

  function renderError(error) {
    if (!error) {
      refs.errorBanner.classList.add("is-hidden");
      return;
    }

    const status = error.status ? `HTTP ${error.status}` : "Frontend";
    const requestId = error.requestId ? ` · Request ${error.requestId}` : "";
    refs.errorBanner.textContent = `${error.code || "ERROR"} · ${status}: ${error.message || "Something went wrong."}${requestId}`;
    refs.errorBanner.classList.remove("is-hidden");
  }

  function renderDebug(requestPayload, responsePayload, errorPayload) {
    refs.requestJson.textContent = requestPayload ? JSON.stringify(requestPayload, null, 2) : "No request sent yet.";
    refs.responseJson.textContent = responsePayload
      ? JSON.stringify(responsePayload, null, 2)
      : errorPayload
        ? JSON.stringify(typeof errorPayload.toJSON === "function" ? errorPayload.toJSON() : errorPayload, null, 2)
        : "No response received yet.";
  }

  return {
    refs,
    renderMediaOptions,
    renderModelOptions,
    renderMediumDetails,
    renderValidation,
    renderResult,
    renderError,
    renderDebug,
    renderIdle,
    setLoading,
  };
}
