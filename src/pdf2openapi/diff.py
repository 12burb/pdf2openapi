"""Compare two versions of an API (PDFs or OpenAPI files) and report what changed."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx
import yaml

from . import schema
from .extract import extract

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_REDIRECTS = 5
ALLOW_PRIVATE_ENV = "PDF2OPENAPI_ALLOW_PRIVATE_URLS"


@dataclass
class Change:
    endpoint: str
    detail: str
    breaking: bool

    def to_dict(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "detail": self.detail, "breaking": self.breaking}


@dataclass
class DiffReport:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)

    @property
    def breaking(self) -> list[Change]:
        return [c for c in self.changes if c.breaking] + [
            Change(e, "endpoint removed", True) for e in self.removed
        ]

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "changes": [c.to_dict() for c in self.changes],
            "breaking_count": len(self.breaking),
        }

    def to_markdown(self) -> str:
        if self.empty:
            return "No API changes detected."
        out = [f"## API changes ({len(self.breaking)} breaking)", ""]
        if self.removed:
            out += ["### Removed endpoints", *[f"- **BREAKING** `{e}`" for e in self.removed], ""]
        if self.added:
            out += ["### New endpoints", *[f"- `{e}`" for e in self.added], ""]
        if self.changes:
            out.append("### Changed endpoints")
            for c in self.changes:
                flag = "**BREAKING** " if c.breaking else ""
                out.append(f"- {flag}`{c.endpoint}`: {c.detail}")
            out.append("")
        return "\n".join(out).rstrip() + "\n"


def _check_public_url(url: str) -> str | None:
    """Refuse URLs that point at loopback, private, or link-local addresses.

    Returns the checked IP address so the request can connect to exactly that address (a second
    DNS lookup could return a private one), or None when private URLs are allowed.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"unsupported URL: {url}")
    if os.environ.get(ALLOW_PRIVATE_ENV) == "1":
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve {parsed.hostname}: {exc}") from exc
    ips = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
    for ip in ips:
        if not ip.is_global:
            raise ValueError(
                f"refusing to fetch {url}: {parsed.hostname} resolves to non-public address {ip} "
                f"(set {ALLOW_PRIVATE_ENV}=1 to allow)"
            )
    if not ips:
        raise ValueError(f"cannot resolve {parsed.hostname}")
    return str(ips[0])


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    return (
        parsed.scheme,
        parsed.hostname or "",
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


def _request_target(url: str, pin: bool) -> tuple[str, dict[str, str], dict[str, Any]]:
    """URL to request (credentials removed, host pinned to the checked IP), headers, extensions."""
    parsed = urlparse(url)
    host = f"[{parsed.hostname}]" if ":" in (parsed.hostname or "") else parsed.hostname or ""
    hostport = host + (f":{parsed.port}" if parsed.port else "")
    headers: dict[str, str] = {}
    extensions: dict[str, Any] = {}
    netloc = hostport
    ip = _check_public_url(url) if pin else None
    if ip:
        netloc = (f"[{ip}]" if ":" in ip else ip) + (f":{parsed.port}" if parsed.port else "")
        headers["Host"] = hostport
        if parsed.scheme == "https":
            extensions["sni_hostname"] = parsed.hostname
    return parsed._replace(netloc=netloc).geturl(), headers, extensions


def fetch_url(url: str, transport: httpx.BaseTransport | None = None) -> bytes:
    """Download a document, checking every redirect hop and capping the size.

    Credentials in the URL are only sent to that URL's own origin, never to redirect targets.
    """
    original = url
    parsed = urlparse(url)
    home = _origin(url)
    auth = None
    if parsed.username is not None:
        auth = httpx.BasicAuth(unquote(parsed.username), unquote(parsed.password or ""))
    with httpx.Client(timeout=60, transport=transport, trust_env=transport is None) as client:
        for _ in range(MAX_REDIRECTS + 1):
            target, headers, extensions = _request_target(url, pin=transport is None)
            request = client.build_request("GET", target, headers=headers, extensions=extensions)
            if auth and _origin(url) == home:
                request = next(auth.auth_flow(request))
            try:
                resp = client.send(request, stream=True)
            except httpx.HTTPError as exc:
                raise ValueError(f"could not fetch {original}: {exc}") from exc
            try:
                if resp.is_redirect:
                    location = urljoin(url, resp.headers["location"])
                    url = (
                        urlparse(location)
                        ._replace(netloc=urlparse(location).netloc.rpartition("@")[2])
                        .geturl()
                    )
                    continue
                if resp.status_code >= 400:
                    raise ValueError(f"{original} returned HTTP {resp.status_code}")
                chunks, size = [], 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                resp.close()
    raise ValueError(f"too many redirects fetching {url}")


def _parse_spec_text(text: str, source: str, is_json: bool) -> dict[str, Any]:
    try:
        data = json.loads(text) if is_json else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        kind = "JSON" if is_json else "YAML"
        raise ValueError(f"{source} is not valid {kind}: {exc}") from exc
    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError(f"{source} is not an OpenAPI document")
    return data


def load_spec(
    source: str, ocr: bool = True, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Load an OpenAPI spec from a .pdf/.yaml/.json path or http(s) URL."""
    if re.match(r"https?://", source):
        content = fetch_url(source, transport=transport)
        if content[:5] == b"%PDF-":
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "download.pdf"
                path.write_bytes(content)
                return extract(path, ocr=ocr).spec
        text = _decode(content, source)
        return _parse_spec_text(text, source, text.lstrip().startswith(("{", "[")))
    path = Path(source)
    with open(path, "rb") as fh:
        head = fh.read(5)
    if head == b"%PDF-":
        return extract(path, ocr=ocr).spec
    return _parse_spec_text(_decode(path.read_bytes(), source), source, path.suffix == ".json")


def _decode(content: bytes, source: str) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} is neither a PDF nor a text (YAML/JSON) file") from exc


def _operations(spec: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    ops = {}
    for path, item in (spec.get("paths") or {}).items():
        for method, op in (item or {}).items():
            if method in HTTP_METHODS and isinstance(op, dict):
                shape = re.sub(r"\{[^}]+\}", "{}", path)
                ops[f"{method.upper()} {shape}"] = (f"{method.upper()} {path}", op)
    return ops


def _params(op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for p in op.get("parameters") or []:
        if isinstance(p, dict) and "name" in p and p.get("in") != "path":
            out[f"{p.get('in')} param '{p['name']}'"] = {
                "type": schema.describe_type(p.get("schema") or {}),
                "required": bool(p.get("required")),
            }
    return out


def _json_schema(container: dict[str, Any] | None) -> dict[str, Any]:
    content = (container or {}).get("content") or {}
    for media in content.values():
        if isinstance(media, dict) and isinstance(media.get("schema"), dict):
            return media["schema"]
    return {}


def _body_fields(op: dict[str, Any]) -> dict[str, dict[str, Any]]:
    body = op.get("requestBody")
    if not isinstance(body, dict):
        return {}
    root = _json_schema(body)
    out = {
        "request body": {"type": schema.describe_type(root), "required": bool(body.get("required"))}
    }
    flat = schema.flatten(root)
    out.update({f"body field '{k}'": v for k, v in flat.items()})
    return out


def _success_response(op: dict[str, Any]) -> dict[str, Any]:
    responses = op.get("responses") or {}
    for code in sorted(responses):
        if str(code).startswith("2"):
            return responses[code]
    return {}


def _compare_inputs(endpoint: str, old: dict, new: dict, changes: list[Change]) -> None:
    for name in sorted(set(old) | set(new)):
        if name not in new:
            changes.append(Change(endpoint, f"{name} removed", True))
        elif name not in old:
            req = new[name]["required"]
            changes.append(
                Change(endpoint, f"{name} added ({'required' if req else 'optional'})", req)
            )
        else:
            o, n = old[name], new[name]
            if o["type"] != n["type"]:
                changes.append(Change(endpoint, f"{name} type {o['type']} -> {n['type']}", True))
            if not o["required"] and n["required"]:
                changes.append(Change(endpoint, f"{name} is now required", True))
            elif o["required"] and not n["required"]:
                changes.append(Change(endpoint, f"{name} is now optional", False))


def _compare_outputs(endpoint: str, old_op: dict, new_op: dict, changes: list[Change]) -> None:
    old_codes = set(map(str, old_op.get("responses") or {}))
    new_codes = set(map(str, new_op.get("responses") or {}))
    for code in sorted(old_codes - new_codes):
        changes.append(Change(endpoint, f"response {code} removed", code.startswith("2")))
    for code in sorted(new_codes - old_codes):
        changes.append(Change(endpoint, f"response {code} added", False))
    old_root = _json_schema(_success_response(old_op))
    new_root = _json_schema(_success_response(new_op))
    if old_root and new_root:
        o, n = schema.describe_type(old_root), schema.describe_type(new_root)
        if o != n:
            changes.append(Change(endpoint, f"response body type {o} -> {n}", True))
    old_fields = schema.flatten(old_root)
    new_fields = schema.flatten(new_root)
    for name in sorted(set(old_fields) | set(new_fields)):
        if name not in new_fields:
            changes.append(Change(endpoint, f"response field '{name}' removed", True))
        elif name not in old_fields:
            changes.append(Change(endpoint, f"response field '{name}' added", False))
        elif old_fields[name]["type"] != new_fields[name]["type"]:
            o, n = old_fields[name]["type"], new_fields[name]["type"]
            changes.append(Change(endpoint, f"response field '{name}' type {o} -> {n}", True))
        elif old_fields[name]["required"] and not new_fields[name]["required"]:
            changes.append(Change(endpoint, f"response field '{name}' is now optional", True))


def _security_names(requirements: Any) -> list[str]:
    """Canonical security requirements: alternatives (OR) of schemes with scopes (AND)."""
    out = []
    for req in requirements or []:
        if isinstance(req, dict):
            parts = [
                f"{name}[{','.join(sorted(scopes))}]" if scopes else name
                for name, scopes in sorted(req.items())
            ]
            out.append(" + ".join(parts))
    return sorted(out)


def _compare_security(old: dict[str, Any], new: dict[str, Any], changes: list[Change]) -> None:
    old_schemes = (old.get("components") or {}).get("securitySchemes") or {}
    new_schemes = (new.get("components") or {}).get("securitySchemes") or {}
    if set(old_schemes) != set(new_schemes):
        changes.append(
            Change(
                "(global)",
                f"auth schemes {sorted(old_schemes)} -> {sorted(new_schemes)}",
                True,
            )
        )
    for name in sorted(set(old_schemes) & set(new_schemes)):
        if old_schemes[name] != new_schemes[name]:
            o = json.dumps(old_schemes[name], sort_keys=True)
            n = json.dumps(new_schemes[name], sort_keys=True)
            changes.append(Change("(global)", f"auth scheme '{name}' changed: {o} -> {n}", True))
    o, n = _security_names(old.get("security")), _security_names(new.get("security"))
    if o != n:
        changes.append(Change("(global)", f"required auth {o} -> {n}", True))


def diff_specs(old: dict[str, Any], new: dict[str, Any]) -> DiffReport:
    report = DiffReport()
    old_ops, new_ops = _operations(old), _operations(new)
    for key in sorted(set(old_ops) | set(new_ops)):
        if key not in new_ops:
            report.removed.append(old_ops[key][0])
            continue
        if key not in old_ops:
            report.added.append(new_ops[key][0])
            continue
        label, new_op = new_ops[key]
        old_op = old_ops[key][1]
        _compare_inputs(label, _params(old_op), _params(new_op), report.changes)
        _compare_inputs(label, _body_fields(old_op), _body_fields(new_op), report.changes)
        _compare_outputs(label, old_op, new_op, report.changes)
        if "security" in old_op or "security" in new_op:
            o = _security_names(old_op.get("security", old.get("security")))
            n = _security_names(new_op.get("security", new.get("security")))
            if o != n:
                report.changes.append(Change(label, f"required auth {o} -> {n}", True))

    _compare_security(old, new, report.changes)
    old_servers = [s.get("url") for s in old.get("servers") or []]
    new_servers = [s.get("url") for s in new.get("servers") or []]
    if old_servers != new_servers:
        report.changes.append(
            Change("(global)", f"servers {old_servers} -> {new_servers}", bool(old_servers))
        )
    return report
