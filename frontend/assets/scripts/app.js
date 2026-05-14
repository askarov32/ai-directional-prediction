import { ApiError, createPrediction, fetchMedia, fetchModels } from "./api.js";
import { renderDomain } from "./charts.js";
import { applyModelDomainPolicy, buildDemoPayload, fillForm, readPayloadFromForm } from "./form.js";
import { getState, setState, subscribe } from "./state.js";
import { createUI } from "./ui.js";
import { validatePayload } from "./validators.js";

const ui = createUI();

function getMediumById(mediumId) {
  return getState().media.find((medium) => medium.id === mediumId) || null;
}

function getModelById(modelId) {
  return getState().models.find((model) => model.id === modelId) || null;
}

function syncDraftFromForm() {
  const rawDraftRequest = readPayloadFromForm(ui.refs.form);
  const model = getModelById(rawDraftRequest.model);
  const draftRequest = applyModelDomainPolicy(rawDraftRequest, model);
  if (JSON.stringify(draftRequest) !== JSON.stringify(rawDraftRequest)) {
    fillForm(ui.refs.form, draftRequest);
  }
  const medium = getMediumById(draftRequest.medium_id);
  const validationErrors = validatePayload(draftRequest, medium, model);

  setState({
    draftRequest,
    selectedMediumId: draftRequest.medium_id,
    selectedModel: draftRequest.model,
    validationErrors,
  });

  return draftRequest;
}

function hydrateFormWithDemo() {
  const state = getState();
  const selectedMediumId = state.selectedMediumId || state.media[0]?.id || "";
  const selectedModel =
    state.models.find((model) => model.id === state.selectedModel) ||
    state.models.find((model) => model.status === "configured" && model.default_domain_type === "rect_3d") ||
    state.models.find((model) => model.status === "configured") ||
    state.models[0] ||
    { id: "meshgraphnet", default_domain_type: "rect_3d", supported_domain_types: ["rect_2d", "rect_3d"] };
  const payload = buildDemoPayload(selectedMediumId, selectedModel);
  fillForm(ui.refs.form, payload);
  syncDraftFromForm();
}

function resetForm() {
  const state = getState();
  const selectedModel =
    state.models.find((model) => model.id === state.selectedModel) ||
    { id: state.selectedModel || "meshgraphnet" };
  const payload = buildDemoPayload(state.selectedMediumId || state.media[0]?.id || "", selectedModel);
  fillForm(ui.refs.form, payload);
  setState({
    lastResponse: null,
    error: null,
    lastRequest: null,
  });
  syncDraftFromForm();
}

function updateView(state) {
  const medium = getMediumById(state.selectedMediumId);
  const model = getModelById(state.selectedModel);

  if (state.media.length) {
    ui.renderMediaOptions(state.media, state.selectedMediumId);
  }

  if (state.models.length) {
    ui.renderModelOptions(state.models, state.selectedModel);
  }

  ui.renderMediumDetails(medium);
  ui.renderValidation(state.validationErrors);
  ui.setLoading(state.loading);
  ui.renderDebug(state.lastRequest, state.lastResponse, state.error);

  if (state.lastResponse) {
    ui.renderResult(state.lastResponse, model?.name || state.selectedModel);
  } else {
    ui.renderIdle(model?.name || state.selectedModel);
  }

  ui.renderError(state.error);

  renderDomain(ui.refs.domainSvg, state.draftRequest, state.lastResponse, model?.name || state.selectedModel);
}

async function handleSubmit(event) {
  event.preventDefault();
  const payload = syncDraftFromForm();
  const medium = getMediumById(payload.medium_id);
  const model = getModelById(payload.model);
  const validationErrors = validatePayload(payload, medium, model);

  if (Object.keys(validationErrors).length > 0) {
    setState({ validationErrors, error: null });
    return;
  }

  setState({
    loading: true,
    error: null,
    lastRequest: payload,
  });

  try {
    const response = await createPrediction(payload);
    setState({
      loading: false,
      lastResponse: response,
      error: null,
    });
  } catch (error) {
    setState({
      loading: false,
      error: error instanceof ApiError ? error : new ApiError({ code: "PREDICTION_FAILED", message: error.message }),
      lastResponse: null,
    });
  }
}

async function bootstrap() {
  subscribe(updateView);

  try {
    const [media, models] = await Promise.all([fetchMedia(), fetchModels()]);
    setState({
      media,
      models,
      selectedMediumId: media[0]?.id || "",
      selectedModel:
        models.find((model) => model.status === "configured" && model.default_domain_type === "rect_3d")?.id ||
        models.find((model) => model.status === "configured")?.id ||
        models[0]?.id ||
        "meshgraphnet",
    });

    hydrateFormWithDemo();
  } catch (error) {
    setState({
      error:
        error instanceof ApiError
          ? error
          : new ApiError({
              code: "BOOTSTRAP_FAILED",
              message: error.message || "Failed to load initial application data.",
            }),
    });
  }
}

ui.refs.form.addEventListener("input", () => {
  syncDraftFromForm();
});

ui.refs.form.addEventListener("change", () => {
  syncDraftFromForm();
});

ui.refs.form.addEventListener("submit", handleSubmit);
ui.refs.demoButton.addEventListener("click", hydrateFormWithDemo);
ui.refs.resetButton.addEventListener("click", resetForm);
ui.refs.copyJsonButton.addEventListener("click", async () => {
  const response = getState().lastResponse;
  if (!response) {
    return;
  }
  try {
    await navigator.clipboard.writeText(JSON.stringify(response, null, 2));
    ui.refs.copyJsonButton.textContent = "Copied";
  } catch (error) {
    ui.refs.copyJsonButton.textContent = "Clipboard blocked";
  }
  window.setTimeout(() => {
    ui.refs.copyJsonButton.textContent = "Copy JSON";
  }, 1200);
});

bootstrap();
