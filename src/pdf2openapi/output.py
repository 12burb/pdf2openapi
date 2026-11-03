"""Serialize specs."""

from __future__ import annotations

import json
from typing import Any

import yaml


def dump(spec: dict[str, Any], fmt: str = "yaml") -> str:
    if fmt == "json":
        return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    return yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100)
