"""Validate a generated spec and optionally probe a vendor sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from openapi_spec_validator import OpenAPIV30SpecValidator, OpenAPIV31SpecValidator

SAFE_METHODS = ("get", "head")


@dataclass
class ProbeResult:
    endpoint: str
    status: int | None
    verdict: str


def validate_spec(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors mean the spec is not valid OpenAPI."""
    version = str(spec.get("openapi", ""))
    validator_cls = (
        OpenAPIV31SpecValidator if version.startswith("3.1") else OpenAPIV30SpecValidator
    )
    errors = []
    for err in validator_cls(spec).iter_errors():
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"{where}: {err.message}")

    warnings = []
    for path, item in (spec.get("paths") or {}).items():
        for method, op in (item or {}).items():
            if not isinstance(op, dict):
                continue
            key = f"{method.upper()} {path}"
            if op.get("x-inferred-responses"):
                warnings.append(f"{key}: no status codes in the docs; assumed 200")
            if method in ("post", "put", "patch") and "requestBody" not in op:
                warnings.append(f"{key}: no request body documented")
    return errors, warnings


def probe(
    spec: dict[str, Any],
    base_url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 15,
    transport: httpx.BaseTransport | None = None,
) -> list[ProbeResult]:
    """Call read-only endpoints that need no inputs, to check the docs match reality."""
    results = []
    with httpx.Client(
        base_url=base_url.rstrip("/"), headers=headers or {}, timeout=timeout, transport=transport
    ) as client:
        for path, item in (spec.get("paths") or {}).items():
            for method, op in (item or {}).items():
                if method not in SAFE_METHODS or not isinstance(op, dict):
                    continue
                key = f"{method.upper()} {path}"
                if not _is_relative_path(path):
                    results.append(
                        ProbeResult(key, None, "skipped (path is not relative to base URL)")
                    )
                    continue
                needs_input = "{" in path or any(
                    p.get("required") for p in op.get("parameters") or [] if p.get("in") != "header"
                )
                if needs_input:
                    results.append(ProbeResult(key, None, "skipped (needs path or query values)"))
                    continue
                try:
                    resp = client.request(method.upper(), path)
                except httpx.HTTPError as exc:
                    results.append(ProbeResult(key, None, f"error: {exc}"))
                    continue
                results.append(ProbeResult(key, resp.status_code, _verdict(resp.status_code, op)))
    return results


def _is_relative_path(path: str) -> bool:
    return (
        path.startswith("/")
        and not path.startswith("//")
        and "://" not in path
        and "\\" not in path
    )


def _verdict(status: int, op: dict[str, Any]) -> str:
    documented = set(map(str, op.get("responses") or {}))
    if 200 <= status < 300:
        return (
            "ok"
            if str(status) in documented or op.get("x-inferred-responses")
            else (f"ok, but {status} is not documented")
        )
    if status in (401, 403):
        return "auth failed (check your headers)"
    if status == 404:
        return "not found: the docs may have the wrong path"
    if status == 405:
        return "method not allowed: the docs may have the wrong method"
    return f"unexpected status {status}"
