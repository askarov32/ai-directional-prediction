import { formatRange } from "./validators.js";

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

export function createUI() {
  const refs = {
    form: document.querySelector("#prediction-form"),
    mediumSelect: document.querySelector("#medium-select"),
    modelSelect: document.querySelector("#model-select"),
    modelHint: document.querySelector("#model-capability-hint"),
    mediumProperties: document.querySelector("#medium-properties"),
    temperatureRangeHint: document.querySelector("#temperature-range-hint"),
    pressureRangeHint: document.querySelector("#pressure-range-hint"),
    predictButton: document.querySelector("#predict-button"),
    demoButton: document.querySelector("#demo-button"),
    resetButton: document.querySelector("#reset-button"),
    resultCard: document.querySelector("#result-card"),
    resultEmpty: document.querySelector("#result-empty"),
    resultContent: document.querySelector("#result-content"),
    errorBanner: document.querySelector("#error-banner"),
    direction: document.querySelector("#result-direction"),
    azimuth: document.querySelector("#result-azimuth"),
    elevation: document.querySelector("#result-elevation"),
    magnitude: document.querySelector("#result-magnitude"),
    travelTime: document.querySelector("#result-travel-time"),
    displacement: document.querySelector("#result-displacement"),
    temperaturePerturbation: document.querySelector("#result-temperature-perturbation"),
    modelVersion: document.querySelector("#result-model-version"),
    requestId: document.querySelector("#result-request-id"),
    latencyBadge: document.querySelector("#latency-badge"),
    modelBadge: document.querySelector("#model-badge"),
    requestJson: document.querySelector("#request-json"),
    responseJson: document.querySelector("#response-json"),
    copyJsonButton: document.querySelector("#copy-json-button"),
    domainSvg: document.querySelector("#domain-svg"),
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
        label:
          model.default_domain_type === "rect_3d"
            ? `${model.name} · 3D ready`
            : model.supported_domain_types?.includes("rect_2d") && !model.supported_domain_types?.includes("rect_3d")
              ? `${model.name} · 2D only`
              : model.name,
      })),
      selectedModel
    );
    const activeModel = models.find((model) => model.id === selectedModel) || null;
    refs.modelHint.textContent = activeModel?.capability_note || "Requests are routed by the FastAPI orchestration layer.";
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
      refs.temperatureRangeHint.textContent = "";
      refs.pressureRangeHint.textContent = "";
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

    refs.temperatureRangeHint.textContent = formatRange(medium.ranges.temperature_c, "°C");
    refs.pressureRangeHint.textContent = formatRange(medium.ranges.pressure_mpa, "MPa");
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

    const unitDir = geom.unit_direction || {};
    refs.direction.textContent =
      Number.isFinite(unitDir.x) && Number.isFinite(unitDir.y)
        ? `[${unitDir.x.toFixed(3)}, ${unitDir.y.toFixed(3)}]`
        : "—";
    refs.azimuth.textContent =
      Number.isFinite(geom.azimuth_deg) ? `${geom.azimuth_deg.toFixed(1)}°` : "—";
    // v2 is 2D plane-strain — elevation is omitted from the contract
    refs.elevation.textContent = "n/a (2D)";
    refs.magnitude.textContent =
      Number.isFinite(dir.response_magnitude_score)
        ? dir.response_magnitude_score.toFixed(3)
        : "—";

    const travelTimeMs =
      Number.isFinite(temporal.travel_time_s)
        ? temporal.travel_time_s * 1000.0
        : NaN;
    refs.travelTime.textContent =
      Number.isFinite(travelTimeMs) ? `${travelTimeMs.toFixed(3)} ms` : "—";

    const magnitudeM = _safeNum(disp.magnitude_m);
    refs.displacement.textContent = Number.isFinite(magnitudeM)
      ? prettyNumber(magnitudeM)
      : "—";

    const thetaK = _safeNum(therm.temperature_perturbation_k?.value);
    refs.temperaturePerturbation.textContent = Number.isFinite(thetaK)
      ? prettyNumber(thetaK)
      : "—";

    refs.modelVersion.textContent = modelMeta.version || "—";
    refs.requestId.textContent = response.request_id || "—";
    const latencyMs = _safeNum(modelMeta.inference_time_ms);
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

    const waveTypeNode = document.querySelector("#result-wave-type");
    if (waveTypeNode) {
      waveTypeNode.textContent = diag.notes?.[0] || "v2 prototype prediction";
    }
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

    // v1 legacy path — byte-identical behaviour
    refs.resultCard.classList.add("is-live");
    refs.resultEmpty.classList.add("is-hidden");
    refs.resultContent.classList.remove("is-hidden");
    refs.errorBanner.classList.add("is-hidden");

    refs.direction.textContent = `[${response.prediction.direction_vector.map((item) => item.toFixed(3)).join(", ")}]`;
    refs.azimuth.textContent = `${response.prediction.azimuth_deg.toFixed(1)}°`;
    refs.elevation.textContent = `${response.prediction.elevation_deg.toFixed(1)}°`;
    refs.magnitude.textContent = response.prediction.magnitude.toFixed(3);
    refs.travelTime.textContent = `${response.prediction.travel_time_ms.toFixed(2)} ms`;
    refs.displacement.textContent = response.field_summary.max_displacement.toFixed(6);
    refs.temperaturePerturbation.textContent = response.field_summary.max_temperature_perturbation.toFixed(6);
    refs.modelVersion.textContent = response.meta.model_version;
    refs.requestId.textContent = response.meta.request_id;
    refs.latencyBadge.textContent = `Latency ${response.meta.latency_ms} ms`;
    refs.modelBadge.textContent = modelLabel || response.model;
    refs.modelBadge.className = "model-pill";
    refs.modelBadge.dataset.model = response.model;
    document.querySelector("#result-wave-type").textContent = response.prediction.wave_type;
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
