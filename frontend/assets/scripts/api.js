const FALLBACK_BASE_URLS =
  window.location.protocol === "file:"
    ? ["http://localhost:8000/api/v1"]
    : [
        window.THESIS_APP_CONFIG?.apiBaseUrl,
        "/api/v1",
        `${window.location.protocol}//${window.location.hostname}:8000/api/v1`,
        "http://localhost:8000/api/v1",
      ].filter(Boolean);

export const API_BASE_URL = FALLBACK_BASE_URLS[0];

async function rawRequest(baseUrl, path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (parseError) {
        data = null;
      }
    }

    if (!response.ok) {
      throw {
        status: response.status,
        code: data?.error?.code || "REQUEST_FAILED",
        message: data?.error?.message || `Request failed with status ${response.status}`,
        details: data?.error?.details || {},
      };
    }

    return data || {};
  } catch (error) {
    if (error.name === "AbortError") {
      throw {
        code: "REQUEST_TIMEOUT",
        message: "The request took too long and was cancelled in the browser.",
        details: {},
      };
    }
    if (error instanceof TypeError) {
      throw {
        code: "NETWORK_ERROR",
        message: "Unable to reach the backend API.",
        details: {},
      };
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function request(path, options = {}) {
  let lastError = null;

  for (const baseUrl of FALLBACK_BASE_URLS) {
    try {
      return await rawRequest(baseUrl, path, options);
    } catch (error) {
      lastError = error;
      if (error?.code !== "NETWORK_ERROR" && error?.code !== "REQUEST_TIMEOUT") {
        throw error;
      }
    }
  }

  throw lastError || {
    code: "NETWORK_ERROR",
    message: "Unable to reach the backend API.",
    details: {},
  };
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
