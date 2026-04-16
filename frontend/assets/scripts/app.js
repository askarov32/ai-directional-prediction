import { createPrediction, fetchMedia, fetchModels } from "./api.js";
import { renderDomain } from "./charts.js";
import { buildDemoPayload, fillForm, readPayloadFromForm } from "./form.js";
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
  const draftRequest = readPayloadFromForm(ui.refs.form);
  const medium = getMediumById(draftRequest.medium_id);
  const validationErrors = validatePayload(draftRequest, medium);

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
    state.selectedModel ||
    state.models.find((model) => model.status === "configured")?.id ||
    state.models[0]?.id ||
    "meshgraphnet";
  const payload = buildDemoPayload(selectedMediumId, selectedModel);
  fillForm(ui.refs.form, payload);
  syncDraftFromForm();
}

function resetForm() {
  const state = getState();
  const payload = buildDemoPayload(state.selectedMediumId || state.media[0]?.id || "", state.selectedModel || "meshgraphnet");
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
  const validationErrors = validatePayload(payload, medium);

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
      error,
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
      selectedModel: models.find((model) => model.status === "configured")?.id || models[0]?.id || "meshgraphnet",
    });

    hydrateFormWithDemo();
  } catch (error) {
    setState({
      error: {
        code: error.code || "BOOTSTRAP_FAILED",
        message: error.message || "Failed to load initial application data.",
        details: error.details || {},
      },
    });
  }
}

ui.refs.form.addEventListener("input", () => {
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
