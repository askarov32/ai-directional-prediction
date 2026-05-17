// Phase 5 (api-contract-v2): the URL flag ?contract=v2 toggles the
// frontend into v2 mode. Default stays v1 until the defense + 1 week
// of green runs, then the default flips here.
const _params = new URLSearchParams(window.location.search);
const _contractRaw = (_params.get("contract") || "").toLowerCase();
export const CONTRACT_VERSION = _contractRaw === "v2" ? "2.0" : "1.0";

const state = {
  media: [],
  models: [],
  selectedMediumId: "",
  selectedModel: "",
  draftRequest: null,
  lastRequest: null,
  lastResponse: null,
  error: null,
  validationErrors: {},
  loading: false,
  contractVersion: CONTRACT_VERSION,
  lastDerivedGeometry: null,
  lastDiagnostics: null,
};

const listeners = new Set();

export function getState() {
  return state;
}

export function setState(patch) {
  Object.assign(state, patch);
  listeners.forEach((listener) => listener({ ...state }));
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
