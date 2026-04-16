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
