const fallbackBaseUrl =
  window.location.protocol === "file:"
    ? "http://localhost:8000/api/v1"
    : `${window.location.protocol}//${window.location.hostname}:8000/api/v1`;

export const API_BASE_URL = window.THESIS_APP_CONFIG?.apiBaseUrl || fallbackBaseUrl;

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const text = await response.text();
    const data = text ? JSON.parse(text) : null;

    if (!response.ok) {
      throw {
        status: response.status,
        code: data?.error?.code || "REQUEST_FAILED",
        message: data?.error?.message || "Request failed",
        details: data?.error?.details || {},
      };
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw {
        code: "REQUEST_TIMEOUT",
        message: "The request took too long and was cancelled in the browser.",
        details: {},
      };
    }
    throw error;
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
