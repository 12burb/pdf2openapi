"""Rule-based conversion of PDF API documentation into an OpenAPI 3.0 spec."""

from __future__ import annotations

import json
import re
from collections import Counter
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
    method_guessed: bool = False
    depth: int | None = None
    closed: bool = False
    truncated: bool = False

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
    pages = [[ln.rstrip() for ln in page.text.splitlines() if ln.strip()] for page in doc.pages]
    running = _running_lines(pages)
    lines: list[tuple[int, str]] = []
    for page, text in zip(doc.pages, pages, strict=True):
        edge = {*range(2), *range(len(text) - 2, len(text))}
        lines.extend(
            (page.number, ln)
            for k, ln in enumerate(text)
            if not (k in edge and _shape(ln) in running)
        )
        lines.extend((page.number, ln.rstrip()) for ln in page.ocr_text.splitlines() if ln.strip())
    return _split_endpoints(_join_wrapped_paths(lines))


def _shape(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _running_lines(pages: list[list[str]]) -> set[str]:
    """Page headers and footers: the same line (ignoring numbers) at the edge of 3+ pages."""
    counts: Counter[str] = Counter()
    for text in pages:
        counts.update({_shape(ln) for ln in text[:2] + text[-2:]})
    return {
        shape
        for shape, n in counts.items()
        if n >= 3
        and not ENDPOINT_RE.search(shape)
        and not _header_columns(shape)
        and not _heading_kind(shape)
    }


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
    r"(?i)^\W*(?:path|uri|url|endpoint|resource(?:\s+url)?|api\s+url|request\s+url)"
    r"(?:\s+(?:to|for|of)\b[^.]{0,40})?\W*$"
)
GUESSED_MARK = "   [method not stated]"
MAX_SECTION_PAGES = 10
PERMISSION_RE = re.compile(r"^[\w-]+:[\w-]+:?[\w-]*(?:\s+[\u221a\u00d7])?$")
GENERIC_HEADING_RE = re.compile(
    r"(?i)^(?:web\s+)?(?:api\s+)?(?:specific\s+)?(?:business\s+)?(?:uri|url|endpoint|overview|introduction|description|"
    r"request|response|specification|parameter|example|sample|function|constraint|"
    r"authori[sz]ation|error|status)s?\b"
)
SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)+\.?|[1-9]\d?)\s+(\S.*)$")


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
    text = NUMBERING_RE.sub("", lines[j][1].strip())
    m = URL_LABEL_RE.match(text)
    if m:
        return m.group(1)
    m = BARE_URL_RE.match(text)
    if m and "/" in m.group(1).split("://")[-1].strip("/"):
        near = lines[max(0, j - 3) : j] + lines[j + 1 : j + 3]
        if any(URL_LABEL_LINE_RE.match(NUMBERING_RE.sub("", t.strip())) for _, t in near):
            return m.group(1)
    return None


def _url_path(url: str) -> str:
    return normalize_path(re.sub(r"^https?://[^/]+", "", url).split("?")[0])


def _explicit_nearby(lines: list[tuple[int, str]], url: str, window: range) -> bool:
    """Is the same path already written as a complete 'METHOD /path' line close by?"""
    target = _url_path(url)
    return any(
        normalize_path(m.group(3)) == target
        for k in window
        for m in ENDPOINT_RE.finditer(lines[k][1])
    )


def _split_endpoints(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Find endpoints whose method and URL sit on separate lines ('Method: GET' / 'Path: /x').

    Each method label is paired with the nearest unclaimed URL line between it and the
    neighbouring method labels, preferring the URL above on ties.
    """
    inserts: dict[int, list[str]] = {}
    seen: dict[str, int] = {}
    claimed: set[int] = set()
    method_at = {i: m for i in range(len(lines)) if (m := _split_method(lines, i))}
    labels = sorted(method_at)
    for n, i in enumerate(labels):
        lo = max(0, i - 4, labels[n - 1] + 1 if n else 0)
        hi = min(len(lines), i + 5, labels[n + 1] if n + 1 < len(labels) else len(lines))
        for j in sorted(range(lo, hi), key=lambda j: (abs(j - i), j > i)):
            url = _split_url(lines, j) if j != i and j not in claimed else None
            if not url:
                continue
            claimed.add(j)
            near = range(max(0, i - 4), min(len(lines), i + 5))
            key = f"{method_at[i]} {url}"
            if not _explicit_nearby(lines, url, near) and i - seen.get(key, -100) > 15:
                inserts.setdefault(min(i, j), []).append(key)
                seen[key] = i
            break
    used = {k.split(" ", 1)[1] for keys in inserts.values() for k in keys}
    for j in range(len(lines)):
        text = NUMBERING_RE.sub("", lines[j][1].strip())
        m = URL_LABEL_RE.match(text)
        url = m.group(1) if m else None
        if (
            not url
            and j > 0
            and URL_LABEL_LINE_RE.match(NUMBERING_RE.sub("", lines[j - 1][1].strip()))
        ):
            b = BARE_URL_RE.match(text)
            url = b.group(1) if b and "/" in b.group(1).split("://")[-1].strip("/") else None
        if not url or url in used or j in claimed or ENDPOINT_RE.search(lines[j][1]):
            continue
        window = range(max(0, j - 4), min(len(lines), j + 5))
        if _explicit_nearby(lines, url, window) or any(k in method_at for k in window):
            continue
        ahead = " ".join(t for _, t in lines[j + 1 : j + 40])
        method = "POST" if re.search(r"(?i)request\s+(body|payload|parameters)", ahead) else "GET"
        inserts.setdefault(j, []).append(f"{method} {url}{GUESSED_MARK}")
        used.add(url)
    if not inserts:
        return lines
    out: list[tuple[int, str]] = []
    for idx, item in enumerate(lines):
        out.extend((item[0], key) for key in inserts.get(idx, []))
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


def _strip_server_prefixes(
    servers: list[str], paths: list[str]
) -> tuple[list[str], dict[str, str], list[str]]:
    """Avoid '/v2/v2/...' when the documented paths repeat the server's path prefix.

    If every path has the prefix, it is dropped from the server. If only some do, the server
    keeps it and the prefix is removed from those paths instead. Returns the servers, path
    renames and warnings.
    """
    out, partial = [], set()
    for url in servers:
        m = re.match(r"(https?://[^/]+)(/.*)?$", url)
        prefix = (m.group(2) or "").rstrip("/") if m else ""
        covered = [p for p in paths if p == prefix or p.startswith(prefix + "/")] if prefix else []
        if covered and len(covered) == len(paths):
            url = m.group(1)
        elif covered:
            partial.add(prefix)
        out.append(url)
    renames: dict[str, str] = {}
    warnings: list[str] = []
    if len(partial) == 1:
        prefix = partial.pop()
        renames = {
            p: p[len(prefix) :] or "/" for p in paths if p == prefix or p.startswith(prefix + "/")
        }
    elif partial:
        warnings.append(
            f"servers have different path prefixes {sorted(partial)} that some paths repeat; "
            "check the server URLs"
        )
    return list(dict.fromkeys(out)), renames, warnings


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
        if URL_LABEL_LINE_RE.match(text) or re.fullmatch(r"(?i)(api\s+)?ur[il]s?", text):
            continue
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


def _heading_depth(line: str) -> int | None:
    """Depth of a numbered heading ('4.1.2 Get order' -> 3), or None for other lines."""
    m = SECTION_NUMBER_RE.match(line)
    if not m or not _is_heading_like(line) or CELL_SPLIT_RE.search(m.group(2)):
        return None
    if len(m.group(2).split()) > 12:
        return None
    return m.group(1).rstrip(".").count(".") + 1


def _find_sections(lines: list[tuple[int, str]]) -> list[Section]:
    sections: list[Section] = []
    headings: list[tuple[str, int]] = []
    for i, (page, line) in enumerate(lines):
        matches = list(ENDPOINT_RE.finditer(line))
        prev = lines[i - 1] if i else None
        if matches and sections and sections[-1].lines[-1:] == [prev] and _is_heading_like(prev[1]):
            sections[-1].lines.pop()
        if matches:
            before = _title_before(lines, i)
            depth = _heading_depth(before)
            titled = [h for h in headings if not GENERIC_HEADING_RE.match(_clean_summary(h[0]))]
            generic = bool(GENERIC_HEADING_RE.match(_clean_summary(before)))
            if titled and (depth is None or generic) and (titled[-1][1] >= 2 or not before):
                before, depth = titled[-1]
            headings = []
        for m in matches:
            guessed = line.endswith(GUESSED_MARK)
            sections.append(
                Section(
                    method=m.group(1),
                    path=normalize_path(m.group(3)),
                    host=m.group(2),
                    page=page,
                    inline=line[m.end() :].removesuffix(GUESSED_MARK),
                    method_guessed=guessed,
                    depth=depth,
                    before=before,
                    query=[
                        k
                        for k in re.findall(r"[?&]([A-Za-z_][\w\-.]*)=", m.group(4) or "")
                        if not k.startswith("_")
                    ],
                )
            )
        if matches:
            continue
        depth = _heading_depth(line)
        if depth:
            headings.append((line, depth))
        if not sections or sections[-1].closed:
            continue
        sec = sections[-1]
        if sec.depth and depth and depth <= sec.depth:
            sec.closed = True
        elif page - sec.page >= MAX_SECTION_PAGES:
            sec.closed = sec.truncated = True
        else:
            sec.lines.append((page, line))
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
    if PERMISSION_RE.match(text) or not re.search(r"[A-Za-z]", text):
        return ""
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


XSD_ROW_RE = re.compile(r"^\s*\d+\s{2,}([A-Za-z][\w]*)\s{2,}(.*?)\bxs:(\w+)(.*)$")
XSD_TYPES = {
    "string": {"type": "string"},
    "date": {"type": "string", "format": "date"},
    "datetime": {"type": "string", "format": "date-time"},
    "boolean": {"type": "boolean"},
    "decimal": {"type": "number"},
    "double": {"type": "number"},
    "float": {"type": "number"},
}


def _parse_xsd_row(line: str) -> tuple[str, dict[str, Any], bool | None, str] | None:
    """Rows of XML-schema tables: 'level  ELEMENT  Description  xs:type  len  min  max  Yes'."""
    m = XSD_ROW_RE.match(line)
    if not m:
        return None
    xs = m.group(3).lower()
    prop = dict(
        XSD_TYPES.get(xs)
        or ({"type": "integer"} if "int" in xs or xs in ("long", "short") else {"type": "string"})
    )
    req = re.search(r"\s(Yes|No|Y|N)\b", m.group(4))
    required = req.group(1).lower().startswith("y") if req else None
    return m.group(1), prop, required, m.group(2).strip()


def _humanize_path(path: str) -> str:
    segs = [s for s in path.strip("/").split("/") if s and not s.startswith("{")]
    segs = [s for s in segs if not re.fullmatch(r"v\d+(\.\d+)*|api", s, re.I)]
    if not segs:
        return ""
    word = re.sub(r"^api[-_]", "", segs[-1], flags=re.I)
    word = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|[-_.]+", " ", word).strip()
    return word[:1].upper() + word[1:]


def _parse_param_line(line: str) -> tuple[str, dict[str, Any], bool | None, str] | None:
    xsd = _parse_xsd_row(line)
    if xsd:
        return xsd
    cells = _cells(line)
    if [c.lower() for c in cells] in (["xml", "json"], ["json", "xml"]):
        return None
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
                if where == "query" and method not in ("GET", "HEAD", "DELETE"):
                    if _parse_xsd_row(line):
                        where = "body"
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
        if not summary and sec.method_guessed:
            summary = _humanize_path(path)
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
    guessed: list[str] = []

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
        if best.truncated:
            warnings.append(
                f"{method} {path}: section cut after {MAX_SECTION_PAGES} pages; "
                "later details may be missing"
            )
        if all(s.method_guessed for s in secs):
            op["x-method-guessed"] = True
            guessed.append(f"{method} {path}")
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
    servers, renames, prefix_warnings = _strip_server_prefixes(servers, list(paths))
    warnings.extend(prefix_warnings)
    for old, new in renames.items():
        for method, op in paths.pop(old).items():
            paths.setdefault(new, {}).setdefault(method, op)
    if servers:
        spec["servers"] = [{"url": s} for s in servers]
    spec["paths"] = dict(sorted(paths.items()))
    if schemes:
        spec["components"] = {"securitySchemes": schemes}
        spec["security"] = [{name: []} for name in schemes]
    if guessed:
        shown = ", ".join(guessed[:3]) + (
            f" and {len(guessed) - 3} more" if len(guessed) > 3 else ""
        )
        warnings.append(
            f"HTTP method not stated near the URL for {len(guessed)} operation(s); guessed "
            f"(marked x-method-guessed): {shown}"
        )
    return ExtractResult(spec=spec, warnings=warnings, sections=sections_text, document=doc)


def extract(path: str | Path, ocr: bool = True) -> ExtractResult:
    return extract_document(pdf.load(path, ocr=ocr))
