"""Command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__, llm
from .diff import diff_specs, load_spec
from .extract import extract
from .output import dump
from .validate import probe, validate_spec


def _warn(lines: list[str]) -> None:
    for line in lines:
        print(f"warning: {line}", file=sys.stderr)


def cmd_extract(args: argparse.Namespace) -> int:
    result = extract(args.pdf, ocr=not args.no_ocr)
    if args.ai or args.provider:
        try:
            provider = llm.resolve_provider(args.provider, args.model)
        except llm.ProviderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Refining with {provider.name}...", file=sys.stderr)

        def progress(done: int, total: int) -> None:
            end = "\n" if done == total else ""
            print(f"\r  {done}/{total} endpoints", end=end, file=sys.stderr, flush=True)

        asyncio.run(llm.enrich(result, provider, progress=progress))
    fmt = args.format or ("json" if args.output and args.output.endswith(".json") else "yaml")
    text = dump(result.spec, fmt)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        count = sum(len(v) for v in result.spec["paths"].values())
        print(f"Wrote {count} endpoints to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    _warn(result.warnings)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    report = diff_specs(
        load_spec(args.old, ocr=not args.no_ocr), load_spec(args.new, ocr=not args.no_ocr)
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sys.stdout.write(report.to_markdown())
    if args.fail_on_breaking and report.breaking:
        return 1
    if args.fail_on_change and not report.empty:
        return 1
    return 0


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers = {}
    for value in values:
        name, sep, val = value.partition(":")
        if not sep:
            raise SystemExit(f"error: header must look like 'Name: value', got {value!r}")
        headers[name.strip()] = val.strip()
    return headers


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    errors, warnings = validate_spec(spec)
    for err in errors:
        print(f"error: {err}")
    _warn(warnings)
    if not errors:
        print("Spec is valid OpenAPI.")
    if args.base_url:
        for r in probe(spec, args.base_url, _parse_headers(args.header)):
            status = r.status if r.status is not None else "-"
            print(f"{status:>4}  {r.endpoint}  {r.verdict}")
    return 1 if errors else 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import server

    server.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2openapi",
        description="Turn vendor PDF API docs into OpenAPI specs and detect when they change.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="convert a PDF into an OpenAPI spec")
    p.add_argument("pdf")
    p.add_argument("-o", "--output", help="write to a file instead of stdout")
    p.add_argument("--format", choices=["yaml", "json"])
    p.add_argument("--no-ocr", action="store_true", help="skip OCR of screenshots and scans")
    p.add_argument("--ai", action="store_true", help="refine with an AI model (bring your own key)")
    p.add_argument("--provider", choices=["openai", "gemini", "ollama"], help="implies --ai")
    p.add_argument("--model", help="model name, e.g. gpt-4o-mini or gemini-2.5-flash")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("diff", help="compare two PDFs or specs (paths or URLs)")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--fail-on-breaking", action="store_true", help="exit 1 on breaking changes")
    p.add_argument("--fail-on-change", action="store_true", help="exit 1 on any change")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("validate", help="validate a spec, optionally against a live sandbox")
    p.add_argument("spec")
    p.add_argument("--base-url", help="call safe GET endpoints on this server")
    p.add_argument("-H", "--header", action="append", default=[], help="'Name: value'")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("mcp", help="run as an MCP server over stdio")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
