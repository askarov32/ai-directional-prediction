export const CONTRACT_VERSION = "2.0";

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
