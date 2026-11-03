"""Helpers for building and flattening JSON schemas."""

from __future__ import annotations

import re
from typing import Any

_TYPES: dict[str, dict[str, Any]] = {
    "string": {"type": "string"},
    "str": {"type": "string"},
    "text": {"type": "string"},
    "char": {"type": "string"},
    "varchar": {"type": "string"},
    "integer": {"type": "integer"},
    "int": {"type": "integer"},
    "int32": {"type": "integer", "format": "int32"},
    "int64": {"type": "integer", "format": "int64"},
    "long": {"type": "integer", "format": "int64"},
    "number": {"type": "number"},
    "float": {"type": "number", "format": "float"},
    "double": {"type": "number", "format": "double"},
    "decimal": {"type": "number"},
    "boolean": {"type": "boolean"},
    "bool": {"type": "boolean"},
    "object": {"type": "object"},
    "dict": {"type": "object"},
    "map": {"type": "object"},
    "json": {"type": "object"},
    "hash": {"type": "object"},
    "array": {"type": "array", "items": {}},
    "list": {"type": "array", "items": {}},
    "date": {"type": "string", "format": "date"},
    "datetime": {"type": "string", "format": "date-time"},
    "date-time": {"type": "string", "format": "date-time"},
    "timestamp": {"type": "string", "format": "date-time"},
    "uuid": {"type": "string", "format": "uuid"},
    "email": {"type": "string", "format": "email"},
    "url": {"type": "string", "format": "uri"},
    "uri": {"type": "string", "format": "uri"},
    "enum": {"type": "string"},
}

_ARRAY_RE = re.compile(r"^(?:array|list)\s*(?:of\s+|<|\[)\s*([a-z\-0-9]+?)s?\s*[>\]]?$")


def parse_type(cell: str) -> dict[str, Any] | None:
    """Map a documentation type label such as 'int64' or 'array of strings' to a schema."""
    word = cell.strip().strip("`*").lower()
    if not word or len(word) > 30:
        return None
    if word in _TYPES:
        return _copy(_TYPES[word])
    if word.endswith("[]"):
        inner = parse_type(word[:-2])
        return {"type": "array", "items": inner or {}} if inner else None
    match = _ARRAY_RE.match(word)
    if match:
        inner = parse_type(match.group(1))
        return {"type": "array", "items": inner or {}} if inner else None
    return None


def _copy(schema: dict[str, Any]) -> dict[str, Any]:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in schema.items()}


def infer(value: Any) -> dict[str, Any]:
    """Infer a schema from an example JSON value."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array", "items": infer(value[0]) if value else {}}
    if isinstance(value, dict):
        return {"type": "object", "properties": {k: infer(v) for k, v in value.items()}}
    return {}


def insert_property(
    root: dict[str, Any], dotted: str, prop: dict[str, Any], required: bool = False
) -> None:
    """Insert 'customer.email' or 'items[].sku' into an object schema, creating parents."""
    parts = [p for p in dotted.split(".") if p]
    node = root
    for i, raw in enumerate(parts):
        is_array = raw.endswith("[]")
        name = raw[:-2] if is_array else raw
        node.setdefault("type", "object")
        props = node.setdefault("properties", {})
        if i == len(parts) - 1:
            existing = props.get(name, {})
            merged = dict(prop)
            if "properties" in existing:
                merged.setdefault("type", "object")
                merged["properties"] = existing["properties"]
            props[name] = merged
            if required:
                req = node.setdefault("required", [])
                if name not in req:
                    req.append(name)
            return
        child = props.setdefault(name, {})
        if is_array:
            child["type"] = "array"
            node = child.setdefault("items", {})
        else:
            child.setdefault("type", "object")
            node = child


def flatten(schema: dict[str, Any] | None, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten an object schema into {'customer.email': {'type': ..., 'required': bool}}."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(schema, dict):
        return out
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        return flatten(schema["items"], prefix + "[]" if prefix else "[]")
    required = set(schema.get("required") or [])
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        key = f"{prefix}.{name}" if prefix else name
        out[key] = {"type": describe_type(prop), "required": name in required}
        out.update(flatten(prop, key))
    return out


def describe_type(schema: dict[str, Any]) -> str:
    kind = schema.get("type", "any")
    if kind == "array":
        items = schema.get("items") or {}
        return f"array<{describe_type(items) if items else 'any'}>"
    fmt = schema.get("format")
    return f"{kind}({fmt})" if fmt else str(kind)
