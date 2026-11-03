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
ENDPOINT_RE = re.compile(
    r"\b(" + "|".join(METHODS) + r")\s+(https?://[^\s/]+)?(/[\w\-./{}:<>~%$]*)(\?[^\s]*)?"
)
WRAPPED_PATH_RE = re.compile(r"(?:/|\{[^}]*|/v\d+(?:\.\d+)?)$")
LABEL_RE = re.compile(
    r"(?i)^(method|url|uri|endpoint|http request|request|response|example|sample request|"
    r"syntax|description|api|resource|request line|definition)(\s+\w+)?\s*:?$"
)
NUMBERING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+")
HEADER_WORDS = {
    "name": {
        "name",
        "parameter",
        "parameter name",
        "param",
        "field",
        "field name",
        "attribute",
        "property",
        "key",
        "element",
    },
    "type": {"type", "data type", "datatype", "format", "value type"},
    "required": {"required", "mandatory", "optional", "req", "req.", "required?", "m/o"},
}
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
    query: list[str] = field(default_factory=list)
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
    path = re.sub(r"\{\$(\w+)\}", r"{\1}", path)
    path = re.sub(r"/:(\w+)", r"/{\1}", path)
    path = re.sub(r"<(?:\w+:)?(\w+)>", r"{\1}", path)
    path = re.sub(r"/{2,}", "/", path)
    path = re.sub(
        r"\{([^}]*)\}", lambda m: "{" + (re.sub(r"\W+", "_", m.group(1)) or "id") + "}", path
    )
    if len(path) > 1:
        path = path.rstrip("/")
    return path or "/"


def _flat_lines(doc: pdf.Document) -> list[tuple[int, str]]:
    lines = [
        (page.number, line.rstrip())
        for page in doc.pages
        for line in page.full_text.splitlines()
        if line.strip()
    ]
    return _split_endpoints(_join_wrapped_paths(lines))


_METHOD_ALT = "|".join(METHODS)
METHOD_LABEL_RE = re.compile(
    r"(?i)^\W*(?:http\s+|request\s+)?(?:method|verb)\s*[:\-|]?\s*(" + _METHOD_ALT + r")\b"
)
BARE_METHOD_RE = re.compile(r"^\W*(" + _METHOD_ALT + r")\W*$")
METHOD_LABEL_LINE_RE = re.compile(
    r"(?i)^\W*(?:http|https|http\s+method|method|verb|request method)\W*$"
)
URL_LABEL_RE = re.compile(
    r"(?i)^\W*(?:path|uri|url|endpoint|resource(?:\s+url)?|api\s+url|request\s+url|"
    r"service\s+url)\s*[:\-|]?\s*((?:https?://[^\s/]+)?/[\w\-./{}:<>~%$?=&]*)\s*$"
)
BARE_URL_RE = re.compile(r"^\W*((?:https?://[^\s/]+)?/[\w\-./{}:<>~%$?=&]*[\w}/])\s*$")
URL_LABEL_LINE_RE = re.compile(
    r"(?i)^\W*(?:path|uri|url|endpoint|resource(?:\s+url)?|api\s+url|request\s+url)\W*$"
)


def _split_method(lines: list[tuple[int, str]], i: int) -> str | None:
    text = lines[i][1].strip()
    m = METHOD_LABEL_RE.match(text)
    if m:
        return m.group(1).upper()
    m = BARE_METHOD_RE.match(text)
    if m and i > 0 and METHOD_LABEL_LINE_RE.match(lines[i - 1][1].strip()):
        return m.group(1).upper()
    return None


def _split_url(lines: list[tuple[int, str]], j: int) -> str | None:
    text = lines[j][1].strip()
    m = URL_LABEL_RE.match(text)
    if m:
        return m.group(1)
    m = BARE_URL_RE.match(text)
    if m and "/" in m.group(1).split("://")[-1].strip("/"):
        near = lines[max(0, j - 3) : j] + lines[j + 1 : j + 3]
        if any(URL_LABEL_LINE_RE.match(t.strip()) for _, t in near):
            return m.group(1)
    return None


def _split_endpoints(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Find endpoints whose method and URL sit on separate lines ('Method: GET' / 'Path: /x')."""
    inserts: dict[int, str] = {}
    seen: dict[str, int] = {}
    for i in range(len(lines)):
        method = _split_method(lines, i)
        if not method:
            continue
        window = range(max(0, i - 4), min(len(lines), i + 5))
        if any(ENDPOINT_RE.search(lines[j][1]) for j in window):
            continue
        for j in sorted(window, key=lambda j: abs(j - i)):
            url = _split_url(lines, j) if j != i else None
            if not url:
                continue
            key = f"{method} {url}"
            if i - seen.get(key, -100) > 15:
                inserts[min(i, j)] = key
                seen[key] = i
            break
    if not inserts:
        return lines
    out: list[tuple[int, str]] = []
    for idx, item in enumerate(lines):
        if idx in inserts:
            out.append((item[0], inserts[idx]))
        out.append(item)
    return out


def _last_match(line: str) -> re.Match[str] | None:
    matches = list(ENDPOINT_RE.finditer(line))
    return matches[-1] if matches else None


def _join_wrapped_paths(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Rejoin endpoint paths that the PDF wrapped onto the next line."""
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        page, line = lines[i]
        m = _last_match(line)
        while (
            m
            and i + 1 < len(lines)
            and m.end() == len(line.rstrip())
            and WRAPPED_PATH_RE.search(m.group(3))
            and re.match(r"[/{$\w]", lines[i + 1][1].lstrip())
            and not ENDPOINT_RE.search(lines[i + 1][1])
        ):
            nxt = lines[i + 1][1].strip()
            head, sep, rest = nxt.partition("   ")
            if not _continues_path(m.group(3), head):
                break
            line = line.rstrip() + head + (sep + rest if sep else "")
            i += 1
            m = _last_match(line)
        out.append((page, line))
        i += 1
    return out


def _continues_path(path: str, head: str) -> bool:
    """Is `head` (start of the next line) the rest of a path the PDF wrapped?"""
    if not re.fullmatch(r"[\w{}$\-.~%:/]+", head):
        return False
    if path.count("{") > path.count("}"):
        return True
    if head.startswith("/"):
        return True
    if not path.endswith("/") or _heading_kind(head) or LABEL_RE.match(head):
        return False
    return "/" in head or "{" in head or head[0].islower() or head[0] in "{$"


def _strip_server_prefixes(servers: list[str], paths: list[str]) -> list[str]:
    """Drop a server's path prefix (e.g. /v2) when the documented paths already include it."""
    out = []
    for url in servers:
        m = re.match(r"(https?://[^/]+)(/.*)?$", url)
        prefix = (m.group(2) or "").rstrip("/") if m else ""
        if m and prefix:
            covered = sum(p == prefix or p.startswith(prefix + "/") for p in paths)
            if paths and covered * 2 >= len(paths):
                url = m.group(1)
        out.append(url)
    return list(dict.fromkeys(out))


def _title_before(lines: list[tuple[int, str]], i: int) -> str:
    """The nearest heading above an endpoint line, skipping labels like 'Method URL'.

    A numbered heading ("2.3 ListMyNumbers") a few lines up wins over a closer unnumbered line,
    which is often a one-sentence description.
    """
    fallback = ""
    for j in range(i - 1, max(i - 7, -1), -1):
        raw = lines[j][1].strip()
        if ENDPOINT_RE.search(raw):
            break
        text = NUMBERING_RE.sub("", raw)
        cells = [c for c in re.split(r"\s{3,}", text) if c]
        if not cells or LABEL_RE.match(text) or all(LABEL_RE.match(c) for c in cells):
            continue
        if not _is_heading_like(raw):
            if fallback or j < i - 3:
                break
            continue
        if NUMBERING_RE.match(raw) and len(cells) == 1:
            return raw
        if not fallback and j >= i - 3:
            fallback = raw
    return fallback


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
                    before=_title_before(lines, i),
                    query=[
                        k
                        for k in re.findall(r"[?&]([A-Za-z_][\w\-.]*)=", m.group(4) or "")
                        if not k.startswith("_")
                    ],
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
    text = NUMBERING_RE.sub("", text)
    text = CELL_SPLIT_RE.split(text.strip())[0] if text.strip() else ""
    if re.match(r"(?i)https?://|[?&/]", text.strip()):
        return ""
    text = text.strip(" \t-:|\u2013\u2014")
    return re.sub(r"\s{2,}", " ", text)


def _is_toc(section: Section) -> bool:
    return bool(re.search(r"\.{4,}\s*\d+\s*$", section.inline)) or not section.lines


def _heading_kind(line: str) -> str | None:
    stripped = NUMBERING_RE.sub("", line.strip()).rstrip(":")
    if len(stripped) > 40 or CELL_SPLIT_RE.search(stripped):
        return None
    for pattern, kind in HEADINGS:
        if pattern.search(stripped):
            return kind
    return None


def _cells(line: str) -> list[str]:
    return [c.strip() for c in CELL_SPLIT_RE.split(line.strip()) if c.strip()]


def _header_columns(line: str) -> dict[str, int] | None:
    """Detect a parameter table header such as 'Parameter | Mandatory | Type | Description'."""
    cells = [c.lower().strip(" :") for c in _cells(line)]
    cols: dict[str, int] = {}
    for idx, cell in enumerate(cells):
        for role, words in HEADER_WORDS.items():
            if cell in words and role not in cols:
                cols[role] = idx
    if "name" in cols and "type" in cols and len(cells) >= 2:
        cols["count"] = len(cells)
        return cols
    return None


def _requiredness(cell: str) -> bool | None:
    word = cell.lower().strip(" .")
    if word in REQUIRED_WORDS:
        return True
    if word in OPTIONAL_WORDS:
        return False
    return None


def _parse_table_row(
    line: str, cols: dict[str, int]
) -> tuple[str, dict[str, Any], bool | None, str] | None:
    cells = _cells(line)
    if len(cells) <= max(cols["name"], cols["type"]):
        return None
    name = cells[cols["name"]].strip("`")
    prop = schema.parse_type(cells[cols["type"]])
    if not NAME_RE.fullmatch(name) or prop is None:
        return None
    required = None
    if "required" in cols and cols["required"] < len(cells):
        required = _requiredness(cells[cols["required"]])
    used = {cols["name"], cols["type"], cols.get("required", -1)}
    desc = " ".join(c for i, c in enumerate(cells) if i not in used)
    if required is None:
        required = _requiredness(cells[-1]) if len(cells) > 2 else None
    return name, prop, required, desc


def _parse_param_line(line: str) -> tuple[str, dict[str, Any], bool | None, str] | None:
    cells = _cells(line)
    if len(cells) >= 2 and NAME_RE.fullmatch(cells[0].strip("`")):
        prop = schema.parse_type(cells[1])
        if prop is not None:
            required = None
            rest = cells[2:]
            if rest and rest[0].lower() in REQUIRED_WORDS | OPTIONAL_WORDS:
                required = rest[0].lower() in REQUIRED_WORDS
                rest = rest[1:]
            if required is None and len(rest) > 1 and _requiredness(rest[-1]) is not None:
                required = _requiredness(rest[-1])
                rest = rest[:-1]
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

        columns: dict[str, int] | None = None
        for idx, line in enumerate(lines):
            for m in STATUS_RE.finditer(line):
                statuses.setdefault(m.group(1), m.group(2).strip())
            header = _header_columns(line)
            if header:
                columns = header
                collecting_desc = False
                continue
            kind = _heading_kind(line)
            parsed = None
            if not kind:
                parsed = (_parse_table_row(line, columns) if columns else None) or (
                    _parse_param_line(line)
                )
            if kind:
                context = kind
                columns = None
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

        for name in sec.query:
            params.setdefault(
                ("query", name), {"name": name, "in": "query", "schema": {"type": "string"}}
            )
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
        if re.fullmatch(r"/v\d+(?:\.\d+)?", path) and not any(len(s.lines) > 3 for s in secs):
            warnings.append(f"{method} {path}: skipped; path looks cut off (page {secs[0].page})")
            continue
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
    servers = _strip_server_prefixes(servers, list(paths))
    if servers:
        spec["servers"] = [{"url": s} for s in servers]
    spec["paths"] = dict(sorted(paths.items()))
    if schemes:
        spec["components"] = {"securitySchemes": schemes}
        spec["security"] = [{name: []} for name in schemes]
    return ExtractResult(spec=spec, warnings=warnings, sections=sections_text, document=doc)


def extract(path: str | Path, ocr: bool = True) -> ExtractResult:
    return extract_document(pdf.load(path, ocr=ocr))
