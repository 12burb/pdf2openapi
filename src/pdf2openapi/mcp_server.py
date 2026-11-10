"""MCP server: lets Claude, Cursor and other MCP clients call pdf2openapi.

When AI refinement is requested and no API key is configured, the server asks the
client's own model through MCP sampling, so the user needs no extra key.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ClientCapabilities, SamplingCapability, SamplingMessage, TextContent

from . import llm, pdf
from .diff import diff_specs, load_spec
from .extract import ai_hint, extract_document
from .output import dump
from .validate import validate_spec

MAX_READ_CHARS = 60000
SPEC_SUFFIXES = (".yaml", ".yml", ".json")

server = FastMCP(
    "pdf2openapi",
    instructions=(
        "Convert vendor PDF API documentation into OpenAPI specs, compare doc versions, "
        "and validate specs. Paths refer to files on the user's machine."
    ),
)


def _parse_pages(pages: str) -> list[int] | None:
    if not pages.strip():
        return None
    out: list[int] = []
    for part in pages.split(","):
        start, _, end = part.strip().partition("-")
        out.extend(range(int(start), int(end or start) + 1))
    return out


def _sampling_provider(ctx: Context) -> llm.CallableProvider:
    async def complete(system: str, prompt: str) -> str:
        result = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            system_prompt=system,
            max_tokens=4000,
            temperature=0,
        )
        content = result.content if isinstance(result.content, list) else [result.content]
        return "".join(getattr(c, "text", "") for c in content)

    return llm.CallableProvider("mcp-client", complete)


def _client_can_sample(ctx: Context) -> bool:
    try:
        return ctx.session.check_client_capability(
            ClientCapabilities(sampling=SamplingCapability())
        )
    except Exception:  # noqa: BLE001
        return False


@server.tool()
async def extract_openapi(
    pdf_path: str,
    use_ai: bool = False,
    output_path: str = "",
    format: str = "yaml",
    ctx: Context | None = None,
) -> str:
    """Convert a vendor PDF API reference into an OpenAPI 3.0 spec.

    Every operation has x-source-pages citing where it came from. Set use_ai to refine
    the result with a model: the server's configured key if any, otherwise your own model
    via MCP sampling. Set output_path to also write the spec to a file.
    """
    if output_path and Path(output_path).suffix.lower() not in SPEC_SUFFIXES:
        raise ValueError("output_path must end in .yaml, .yml or .json")
    result = extract_document(pdf.load(pdf_path))
    if use_ai:
        provider = None
        if llm.configured_provider_name():
            provider = llm.resolve_provider()
        elif ctx is not None and _client_can_sample(ctx):
            provider = _sampling_provider(ctx)
        if provider:
            await llm.enrich(result, provider)
        else:
            result.warnings.append(
                "AI refinement skipped: no API key configured and this client does not support "
                "sampling. Use read_pdf_pages to review the source pages yourself."
            )
    text = dump(result.spec, "json" if format == "json" else "yaml")
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    header = "".join(f"# warning: {w}\n" for w in result.warnings)
    hint = None if use_ai else ai_hint(result.spec)
    if hint:
        header += f"# tip: {hint} Call again with use_ai=true, or check with read_pdf_pages.\n"
    return header + text


@server.tool()
def read_pdf_pages(pdf_path: str, pages: str = "") -> str:
    """Return the text of a PDF (including OCR of screenshots), marked by page number.

    pages: optional list like "3,5-7". Use this to check or fix an extracted spec.
    """
    text = pdf.page_text(pdf.load(pdf_path), _parse_pages(pages))
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n\n[truncated: request fewer pages]"
    return text


@server.tool()
def diff_api_docs(old: str, new: str, format: str = "markdown") -> str:
    """Compare two API versions. old/new can be PDF, YAML or JSON paths, or URLs.

    Reports added, removed and changed endpoints and flags breaking changes.
    """
    report = diff_specs(load_spec(old), load_spec(new))
    if format == "json":
        return json.dumps(report.to_dict(), indent=2)
    return report.to_markdown()


@server.tool()
def validate_openapi(spec_path: str) -> str:
    """Check that a spec is valid OpenAPI and list gaps worth reviewing."""
    errors, warnings = validate_spec(load_spec(spec_path))
    lines = ["Valid OpenAPI." if not errors else f"{len(errors)} errors:"]
    lines += [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "Warnings:", *[f"- {w}" for w in warnings]]
    return "\n".join(lines)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
