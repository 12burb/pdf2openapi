"""Compare two versions of an API (PDFs or OpenAPI files) and report what changed."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import schema
from .extract import extract

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


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


def load_spec(source: str, ocr: bool = True) -> dict[str, Any]:
    """Load an OpenAPI spec from a .pdf/.yaml/.json path or http(s) URL."""
    if re.match(r"https?://", source):
        resp = httpx.get(source, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        is_pdf = resp.content[:5] == b"%PDF-"
        suffix = ".pdf" if is_pdf else Path(source.split("?")[0]).suffix or ".yaml"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(resp.content)
        source = fh.name
    path = Path(source)
    with open(path, "rb") as fh:
        head = fh.read(5)
    if head == b"%PDF-":
        return extract(path, ocr=ocr).spec
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError(f"{source} is not an OpenAPI document")
    return data


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
    flat = schema.flatten(_json_schema(op.get("requestBody")))
    return {f"body field '{k}'": v for k, v in flat.items()}


def _success_response(op: dict[str, Any]) -> dict[str, Any]:
    responses = op.get("responses") or {}
    for code in sorted(responses):
        if str(code).startswith("2"):
            return responses[code]
    return {}


def _compare_inputs(endpoint: str, old: dict, new: dict, changes: list[Change]) -> None:
    for name in sorted(set(old) | set(new)):
        if name not in new:
            changes.append(Change(endpoint, f"{name} removed", False))
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
    old_fields = schema.flatten(_json_schema(_success_response(old_op)))
    new_fields = schema.flatten(_json_schema(_success_response(new_op)))
    for name in sorted(set(old_fields) | set(new_fields)):
        if name not in new_fields:
            changes.append(Change(endpoint, f"response field '{name}' removed", True))
        elif name not in old_fields:
            changes.append(Change(endpoint, f"response field '{name}' added", False))
        elif old_fields[name]["type"] != new_fields[name]["type"]:
            o, n = old_fields[name]["type"], new_fields[name]["type"]
            changes.append(Change(endpoint, f"response field '{name}' type {o} -> {n}", True))


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

    old_auth = set((old.get("components") or {}).get("securitySchemes") or {})
    new_auth = set((new.get("components") or {}).get("securitySchemes") or {})
    if old_auth != new_auth:
        report.changes.append(
            Change("(global)", f"auth schemes {sorted(old_auth)} -> {sorted(new_auth)}", True)
        )
    old_servers = [s.get("url") for s in old.get("servers") or []]
    new_servers = [s.get("url") for s in new.get("servers") or []]
    if old_servers != new_servers:
        report.changes.append(
            Change("(global)", f"servers {old_servers} -> {new_servers}", bool(old_servers))
        )
    return report
