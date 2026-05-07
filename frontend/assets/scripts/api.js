const DEFAULT_API_BASE_URL = window.location.protocol === "file:" ? "http://localhost:8000/api/v1" : "/api/v1";
const REQUEST_TIMEOUT_MS = Number(window.THESIS_APP_CONFIG?.requestTimeoutMs || 15000);

export const API_BASE_URL = normalizeBaseUrl(window.THESIS_APP_CONFIG?.apiBaseUrl || DEFAULT_API_BASE_URL);

export class ApiError extends Error {
  constructor({ status = 0, code = "REQUEST_FAILED", message = "Request failed.", details = {}, requestId = null }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }

  toJSON() {
    return {
      name: this.name,
      status: this.status,
      code: this.code,
      message: this.message,
      requestId: this.requestId,
      details: this.details,
    };
  }
}

function normalizeBaseUrl(value) {
  return String(value || "").replace(/\/$/, "");
}

function buildRequestId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `frontend-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function parseErrorMessage(code, fallbackMessage) {
  const messages = {
    MODEL_TIMEOUT: "The selected model service timed out. Try again or choose another model.",
    MODEL_UNAVAILABLE: "The selected model service is unavailable. Check Docker readiness or the model host URL.",
    MODEL_HTTP_ERROR: "The selected model service returned an error.",
    MALFORMED_MODEL_RESPONSE: "The selected model returned an invalid response shape.",
    CHECKPOINT_NOT_READY: "PINN checkpoint is not ready yet. Wait for the service or check the checkpoint path.",
    VALIDATION_ERROR: "Some input values are invalid. Review the highlighted fields and debug details.",
    TEMPERATURE_OUT_OF_RANGE: "Temperature is outside the selected medium range.",
    PRESSURE_OUT_OF_RANGE: "Pressure is outside the selected medium range.",
    REQUEST_TIMEOUT: "The backend request timed out.",
    NETWORK_ERROR: "Unable to reach the backend API. Check that Docker Compose is running and nginx proxies /api.",
  };
  return messages[code] || fallbackMessage || "Request failed.";
}

async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ApiError({
      status: response.status,
      code: "INVALID_JSON_RESPONSE",
      message: "Backend returned a non-JSON response.",
      requestId: response.headers.get("X-Request-ID"),
    });
  }
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const requestId = buildRequestId();

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": requestId,
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const data = await parseJsonResponse(response);
    const responseRequestId = response.headers.get("X-Request-ID") || requestId;

    if (!response.ok) {
      const code = data?.error?.code || "REQUEST_FAILED";
      throw new ApiError({
        status: response.status,
        code,
        message: parseErrorMessage(code, data?.error?.message || `Request failed with status ${response.status}`),
        details: data?.error?.details || {},
        requestId: responseRequestId,
      });
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error.name === "AbortError") {
      throw new ApiError({
        code: "REQUEST_TIMEOUT",
        message: parseErrorMessage("REQUEST_TIMEOUT"),
        requestId,
      });
    }
    if (error instanceof TypeError) {
      throw new ApiError({
        code: "NETWORK_ERROR",
        message: parseErrorMessage("NETWORK_ERROR"),
        requestId,
      });
    }
    throw new ApiError({
      code: "UNEXPECTED_FRONTEND_ERROR",
      message: error?.message || "Unexpected frontend request failure.",
      requestId,
    });
  } finally {
    window.clearTimeout(timeout);
  }
}

export function fetchMedia() {
  return request("/media");
}

export function fetchModels() {
  return request("/models");
}

export function createPrediction(payload) {
  return request("/predictions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
