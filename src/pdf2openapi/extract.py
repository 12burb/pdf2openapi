"""Rule-based conversion of PDF API documentation into an OpenAPI 3.0 spec."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any

from . import pdf, schema

METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
ENDPOINT_RE = re.compile(r"\b(" + "|".join(METHODS) + r")\s+(https?://[^\s/]+)?(/[\w\-./{}:<>~%]*)")
NAME_RE = re.compile(r"[A-Za-z_][\w\-]*(?:\[\])?(?:\.[A-Za-z_][\w\-]*(?:\[\])?)*")
CELL_SPLIT_RE = re.compile(r"\s{3,}|\s*\|\s*|\t")
INLINE_PARAM_RE = re.compile(
    r"^[-*\u2022]?\s*`?(" + NAME_RE.pattern + r")`?\s*\(\s*([^,()]+?)\s*"
    r"(?:,\s*(required|optional)\s*)?\)\s*[-:\u2013\u2014]?\s*(.*)$",
    re.IGNORECASE,
)
REQUIRED_WORDS = {"required", "yes", "y", "true", "mandatory", "\u2713", "\u2714", "x"}
OPTIONAL_WORDS = {"optional", "no", "n", "false", "-"}
STATUS_RE = re.compile(
    r"\b([1-5]\d{2})\b\s*[-:]?\s*("
    + "|".join(sorted({re.escape(s.phrase) for s in HTTPStatus}, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
BASE_URL_RE = re.compile(r"(?i)base\s*url\s*[:\-]?\s*(https?://[^\s'\"<>]+)")
VERSION_RE = re.compile(r"(?i)\b(?:api\s+)?version\s*[:\-]?\s*v?(\d+(?:\.\d+)*)")
API_KEY_HEADER_RE = re.compile(r"\b(X-[A-Za-z0-9\-]*(?:Key|Token|Auth)[A-Za-z0-9\-]*)\b")
API_KEY_QUERY_RE = re.compile(r"[?&](api_key|apikey|api-key|access_token)=", re.IGNORECASE)
BEARER_RE = re.compile(r"(?i)authorization\s*:\s*bearer|bearer\s+token")
BASIC_RE = re.compile(r"(?i)basic\s+auth|authorization\s*:\s*basic")

HEADINGS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\b(path|url|route)\s+(param|variable|argument)"), "path"),
    (re.compile(r"(?i)\bquery(\s+string)?\s*(param|argument)?"), "query"),
    (re.compile(r"(?i)\bheaders?\b"), "header"),
    (re.compile(r"(?i)\bresponse|returns\b"), "response"),
    (
        re.compile(r"(?i)\b(request\s+)?(body|payload|form)\b|\brequest\s+(fields|attributes)"),
        "body",
    ),
    (re.compile(r"(?i)\b(request\s+)?param(eter)?s\b|\barguments\b|\bfields\b"), "request"),
    (re.compile(r"(?i)\bexample"), "example"),
]


@dataclass
class Section:
    method: str
    path: str
    host: str | None
    page: int
    inline: str
    before: str
    lines: list[tuple[int, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


@dataclass
class ExtractResult:
    spec: dict[str, Any]
    warnings: list[str]
    sections: dict[str, str]
    document: pdf.Document | None = None


def normalize_path(raw: str) -> str:
    path = raw.split("?")[0].split("#")[0].rstrip(".,:;")
    path = re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{\1}", path)
    path = re.sub(r"/:(\w+)", r"/{\1}", path)
    path = re.sub(r"<(?:\w+:)?(\w+)>", r"{\1}", path)
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


def _flat_lines(doc: pdf.Document) -> list[tuple[int, str]]:
    return [
        (page.number, line.rstrip())
        for page in doc.pages
        for line in page.full_text.splitlines()
        if line.strip()
    ]


def _find_sections(lines: list[tuple[int, str]]) -> list[Section]:
    sections: list[Section] = []
    for i, (page, line) in enumerate(lines):
        matches = list(ENDPOINT_RE.finditer(line))
        if matches and sections and sections[-1].lines and _is_heading_like(lines[i - 1][1]):
            sections[-1].lines.pop()
        for m in matches:
            sections.append(
                Section(
                    method=m.group(1),
                    path=normalize_path(m.group(3)),
                    host=m.group(2),
                    page=page,
                    inline=line[m.end() :],
                    before=lines[i - 1][1] if i > 0 else "",
                )
            )
        if not matches and sections:
            sections[-1].lines.append((page, line))
    return sections


def _is_heading_like(line: str) -> bool:
    """A short title line right before an endpoint belongs to that endpoint, not the previous."""
    line = line.strip()
    return 0 < len(line) <= 80 and not line.endswith(".") and not ENDPOINT_RE.search(line)


def _clean_summary(text: str) -> str:
    text = re.sub(r"\.{3,}\s*\d*\s*$", "", text)
    text = text.strip(" \t-:|\u2013\u2014")
    return re.sub(r"\s{2,}", " ", text)


def _is_toc(section: Section) -> bool:
    return bool(re.search(r"\.{4,}\s*\d+\s*$", section.inline)) or not section.lines


def _heading_kind(line: str) -> str | None:
    stripped = line.strip().rstrip(":")
    if len(stripped) > 40 or CELL_SPLIT_RE.search(stripped):
        return None
    for pattern, kind in HEADINGS:
        if pattern.search(stripped):
            return kind
    return None


def _parse_param_line(line: str) -> tuple[str, dict[str, Any], bool | None, str] | None:
    cells = [c.strip() for c in CELL_SPLIT_RE.split(line.strip()) if c.strip()]
    if len(cells) >= 2 and NAME_RE.fullmatch(cells[0].strip("`")):
        prop = schema.parse_type(cells[1])
        if prop is not None:
            required = None
            rest = cells[2:]
            if rest and rest[0].lower() in REQUIRED_WORDS | OPTIONAL_WORDS:
                required = rest[0].lower() in REQUIRED_WORDS
                rest = rest[1:]
            desc = " ".join(rest)
            if required is None and re.search(r"(?i)\brequired\b", desc):
                required = True
            return cells[0].strip("`"), prop, required, desc
    m = INLINE_PARAM_RE.match(line.strip())
    if m:
        prop = schema.parse_type(m.group(2))
        if prop is not None:
            req = m.group(3)
            return m.group(1), prop, (req.lower() == "required") if req else None, m.group(4)
    return None


def _json_blocks(lines: list[str]) -> list[tuple[int, Any]]:
    """Find JSON objects/arrays in a list of lines; return (start line index, value)."""
    found = []
    i = 0
    while i < len(lines):
        line = lines[i]
        start = min((p for p in (line.find("{"), line.find("[")) if p >= 0), default=-1)
        if start < 0:
            i += 1
            continue
        buf = line[start:]
        depth = 0
        end_line = i
        for j in range(i, min(i + 200, len(lines))):
            chunk = buf if j == i else lines[j]
            if j > i:
                buf += "\n" + chunk
            depth += chunk.count("{") + chunk.count("[") - chunk.count("}") - chunk.count("]")
            end_line = j
            if depth <= 0:
                break
        text = buf.strip().rstrip("'\"")
        text = text[: max(text.rfind("}"), text.rfind("]")) + 1]
        try:
            value = json.loads(text)
        except ValueError:
            i += 1
            continue
        if isinstance(value, (dict, list)) and value:
            found.append((i, value))
            i = end_line + 1
        else:
            i += 1
    return found


def _merge_schema(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    if not base:
        return extra
    if base.get("type") == "object" and extra.get("type") == "object":
        props = dict(extra.get("properties") or {})
        for k, v in (base.get("properties") or {}).items():
            props[k] = _merge_schema(props.get(k, {}), v) if k in props else v
        merged = {**extra, **base, "properties": props}
        req = list(dict.fromkeys((extra.get("required") or []) + (base.get("required") or [])))
        if req:
            merged["required"] = req
        return merged
    return base


class _Builder:
    def __init__(self, auth_headers: set[str]):
        self.auth_headers = {h.lower() for h in auth_headers | {"authorization"}}

    def operation(self, sec: Section, method: str, path: str) -> dict[str, Any]:
        path_params = re.findall(r"\{(\w+)\}", path)
        params: dict[tuple[str, str], dict[str, Any]] = {}
        body: dict[str, Any] = {}
        resp_fields: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        context: str | None = None
        lines = [line for _, line in sec.lines]
        heading_at: dict[int, str] = {}
        description: list[str] = []
        collecting_desc = True

        for idx, line in enumerate(lines):
            for m in STATUS_RE.finditer(line):
                statuses.setdefault(m.group(1), m.group(2).strip())
            kind = _heading_kind(line)
            parsed = None if kind else _parse_param_line(line)
            if kind:
                context = kind
                heading_at[idx] = kind
                collecting_desc = False
                continue
            if parsed:
                collecting_desc = False
                name, prop, required, desc = parsed
                where = self._location(context, name, method, path_params)
                if desc:
                    prop["description"] = desc
                if where == "response":
                    schema.insert_property(resp_fields, name, prop, bool(required))
                elif where == "body":
                    schema.insert_property(body, name, prop, bool(required))
                elif where == "header" and name.lower() in self.auth_headers:
                    continue
                else:
                    param = {"name": name, "in": where, "schema": prop}
                    desc_text = prop.pop("description", None)
                    if desc_text:
                        param["description"] = desc_text
                    if where == "path" or required:
                        param["required"] = True
                    params[(where, name)] = param
            elif (
                collecting_desc
                and len(description) < 3
                and not line.lstrip().startswith(("{", "[", "curl"))
            ):
                description.append(line.strip())

        for name in path_params:
            params.setdefault(
                ("path", name),
                {"name": name, "in": "path", "required": True, "schema": {"type": "string"}},
            )

        request_example = response_example = None
        for start, value in _json_blocks(lines):
            prior = [heading_at[k] for k in sorted(heading_at) if k <= start]
            nearby = " ".join(lines[max(0, start - 2) : start + 1]).lower()
            is_response = (prior and prior[-1] == "response") or "response" in nearby
            if is_response and response_example is None:
                response_example = value
            elif (
                not is_response
                and request_example is None
                and method not in ("GET", "HEAD", "DELETE")
            ):
                request_example = value

        op: dict[str, Any] = {}
        summary = _clean_summary(sec.inline) or self._summary_from_before(sec.before)
        if summary:
            op["summary"] = summary
        if description:
            op["description"] = " ".join(description)
        if params:
            op["parameters"] = list(params.values())

        body_schema = body
        if request_example is not None:
            body_schema = _merge_schema(body, schema.infer(request_example))
        if body_schema:
            content: dict[str, Any] = {"schema": body_schema}
            if request_example is not None:
                content["example"] = request_example
            op["requestBody"] = {
                "required": True,
                "content": {"application/json": content},
            }

        resp_schema = resp_fields
        if response_example is not None:
            resp_schema = _merge_schema(resp_fields, schema.infer(response_example))
        responses: dict[str, Any] = {}
        success = sorted(c for c in statuses if c.startswith("2")) or ["200"]
        for code in sorted(set(statuses) | set(success)):
            entry: dict[str, Any] = {"description": statuses.get(code, "Successful response")}
            if code == success[0] and resp_schema:
                media: dict[str, Any] = {"schema": resp_schema}
                if response_example is not None:
                    media["example"] = response_example
                entry["content"] = {"application/json": media}
            responses[code] = entry
        op["responses"] = responses
        if not statuses:
            op["x-inferred-responses"] = True
        return op

    @staticmethod
    def _summary_from_before(before: str) -> str:
        return _clean_summary(before) if _is_heading_like(before) else ""

    @staticmethod
    def _location(context: str | None, name: str, method: str, path_params: list[str]) -> str:
        if context in ("path", "query", "header", "body", "response"):
            return context
        if name in path_params:
            return "path"
        return "query" if method in ("GET", "HEAD", "DELETE") else "body"


def _operation_id(method: str, path: str, used: set[str]) -> str:
    words = []
    for seg in path.strip("/").split("/"):
        if re.fullmatch(r"v\d+", seg):
            continue
        if seg.startswith("{"):
            words.append("by_" + seg.strip("{}"))
        else:
            words.extend(re.split(r"[^A-Za-z0-9]+", seg))
    words = [w for w in words if w]
    base = method.lower() + "".join(w[:1].upper() + w[1:] for w in words)
    op_id, n = base, 2
    while op_id in used:
        op_id, n = f"{base}{n}", n + 1
    used.add(op_id)
    return op_id


def _title(doc: pdf.Document, lines: list[tuple[int, str]]) -> str:
    if doc.title:
        return doc.title
    for page, line in lines:
        if page != 1:
            break
        if 3 <= len(line.strip()) <= 100:
            return line.strip()
    return Path(doc.path).stem


def _security(full_text: str) -> tuple[dict[str, Any], set[str]]:
    schemes: dict[str, Any] = {}
    headers: set[str] = set()
    header = API_KEY_HEADER_RE.search(full_text)
    if header:
        schemes["apiKeyHeader"] = {"type": "apiKey", "in": "header", "name": header.group(1)}
        headers.add(header.group(1))
    query = API_KEY_QUERY_RE.search(full_text)
    if query:
        schemes["apiKeyQuery"] = {"type": "apiKey", "in": "query", "name": query.group(1)}
    if BEARER_RE.search(full_text):
        schemes["bearerAuth"] = {"type": "http", "scheme": "bearer"}
    if BASIC_RE.search(full_text):
        schemes["basicAuth"] = {"type": "http", "scheme": "basic"}
    return schemes, headers


def extract_document(doc: pdf.Document) -> ExtractResult:
    lines = _flat_lines(doc)
    full_text = "\n".join(line for _, line in lines)
    warnings: list[str] = []

    schemes, auth_headers = _security(full_text)
    builder = _Builder(auth_headers)

    grouped: dict[tuple[str, str], list[Section]] = {}
    for sec in _find_sections(lines):
        grouped.setdefault((sec.method, sec.path), []).append(sec)

    servers = list(dict.fromkeys(m.group(1).rstrip("/.,") for m in BASE_URL_RE.finditer(full_text)))
    for secs in grouped.values():
        for sec in secs:
            if (
                sec.host
                and sec.host not in servers
                and not any(s.startswith(sec.host) for s in servers)
            ):
                servers.append(sec.host)

    paths: dict[str, dict[str, Any]] = {}
    sections_text: dict[str, str] = {}
    used_ids: set[str] = set()
    for (method, path), secs in grouped.items():
        main = [s for s in secs if not _is_toc(s)]
        best = max(main or secs, key=lambda s: len(s.lines))
        op = builder.operation(best, method, path)
        if not op.get("summary"):
            for s in secs:
                summary = _clean_summary(s.inline) or builder._summary_from_before(s.before)
                if summary:
                    op["summary"] = summary
                    break
        source_pages = sorted(
            {p for s in (main or secs) for p in [s.page, *(pg for pg, _ in s.lines)]}
        )
        op["operationId"] = _operation_id(method, path, used_ids)
        op["x-source-pages"] = source_pages
        key = f"{method} {path}"
        sections_text[key] = f"{method} {path}{best.inline}\n{best.text}"
        if not main:
            warnings.append(
                f"{key}: only mentioned in passing (page {best.page}); details may be missing"
            )
        elif (
            "parameters" not in op
            and "requestBody" not in op
            and method in ("POST", "PUT", "PATCH")
        ):
            warnings.append(
                f"{key}: no request fields found (pages {source_pages[0]}-{source_pages[-1]})"
            )
        paths.setdefault(path, {})[method.lower()] = op

    if not paths:
        warnings.append("no endpoints found (looked for lines like 'GET /v1/things')")
    if doc.pages and not doc.ocr_used and not pdf.ocr_available():
        if any(len(p.text.strip()) < pdf.SCANNED_PAGE_CHARS for p in doc.pages):
            warnings.append("some pages have no text layer; install tesseract to OCR them")

    version = VERSION_RE.search(full_text)
    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {
            "title": _title(doc, lines),
            "version": version.group(1) if version else "0.0.0",
            "description": f"Generated by pdf2openapi from {Path(doc.path).name}.",
        },
    }
    if servers:
        spec["servers"] = [{"url": s} for s in servers]
    spec["paths"] = dict(sorted(paths.items()))
    if schemes:
        spec["components"] = {"securitySchemes": schemes}
        spec["security"] = [{name: []} for name in schemes]
    return ExtractResult(spec=spec, warnings=warnings, sections=sections_text, document=doc)


def extract(path: str | Path, ocr: bool = True) -> ExtractResult:
    return extract_document(pdf.load(path, ocr=ocr))
