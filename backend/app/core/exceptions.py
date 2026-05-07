from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ResourceNotFoundError(AppError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=404, details=details or {})


class DomainValidationError(AppError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code=code, message=message, status_code=422, details=details or {})


class RemoteServiceUnavailableError(AppError):
    def __init__(self, service_name: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="MODEL_UNAVAILABLE",
            message=f"{service_name} service is unavailable",
            status_code=503,
            details=details or {},
        )


class RemoteServiceTimeoutError(AppError):
    def __init__(self, service_name: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="MODEL_TIMEOUT",
            message=f"{service_name} service timed out",
            status_code=504,
            details=details or {},
        )


class RemoteServiceHTTPError(AppError):
    def __init__(self, service_name: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="MODEL_HTTP_ERROR",
            message=f"{service_name} service returned an unsuccessful HTTP response",
            status_code=502,
            details=details or {},
        )


class MalformedRemoteResponseError(AppError):
    def __init__(self, service_name: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="MALFORMED_MODEL_RESPONSE",
            message=f"{service_name} service returned a malformed response",
            status_code=502,
            details=details or {},
        )
