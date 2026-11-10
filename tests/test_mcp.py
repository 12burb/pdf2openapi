from __future__ import annotations

import asyncio
import json

import yaml
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CreateMessageResult, TextContent

from pdf2openapi.mcp_server import server


def _text(result) -> str:
    assert not result.isError, result.content
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


def _strip_warnings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("# warning"))


def test_lists_tools():
    async def run():
        async with create_connected_server_and_client_session(server) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert asyncio.run(run()) == {
        "extract_openapi",
        "read_pdf_pages",
        "diff_api_docs",
        "validate_openapi",
    }


def test_extract_read_diff_validate(tmp_path, vendor_pdf, vendor_pdf_v2):
    out = tmp_path / "spec.yaml"

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            spec = await client.call_tool(
                "extract_openapi", {"pdf_path": str(vendor_pdf), "output_path": str(out)}
            )
            pages = await client.call_tool(
                "read_pdf_pages", {"pdf_path": str(vendor_pdf), "pages": "3"}
            )
            diff = await client.call_tool(
                "diff_api_docs", {"old": str(out), "new": str(vendor_pdf_v2), "format": "json"}
            )
            valid = await client.call_tool("validate_openapi", {"spec_path": str(out)})
            return _text(spec), _text(pages), _text(diff), _text(valid)

    spec, pages, diff, valid = asyncio.run(run())
    assert "/v2/orders" in yaml.safe_load(_strip_warnings(spec))["paths"]
    assert out.exists()
    assert pages.startswith("--- page 3 ---") and "GET /v2/orders/:order_id" in pages
    assert json.loads(diff)["added"] == ["GET /v2/refunds"]
    assert valid.startswith("Valid OpenAPI.")


def test_use_ai_falls_back_to_client_sampling(vendor_pdf, monkeypatch):
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PDF2OPENAPI_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    calls = []

    async def sampling(context, params):
        calls.append(params.messages[0].content.text)
        reply = json.dumps({"summary": "From the client model"})
        return CreateMessageResult(
            role="assistant", content=TextContent(type="text", text=reply), model="host-model"
        )

    async def run():
        async with create_connected_server_and_client_session(
            server, sampling_callback=sampling
        ) as client:
            return _text(
                await client.call_tool(
                    "extract_openapi", {"pdf_path": str(vendor_pdf), "use_ai": True}
                )
            )

    spec = yaml.safe_load(_strip_warnings(asyncio.run(run())))
    op = spec["paths"]["/v2/orders"]["post"]
    assert op["summary"] == "From the client model"
    assert op["x-refined-by"] == "mcp-client"
    assert len(calls) == 2


def test_extract_refuses_non_spec_output_path(tmp_path, vendor_pdf):
    async def run():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(
                "extract_openapi",
                {"pdf_path": str(vendor_pdf), "output_path": str(tmp_path / ".bashrc")},
            )

    result = asyncio.run(run())
    assert result.isError and "must end in .yaml" in result.content[0].text
    assert not (tmp_path / ".bashrc").exists()
