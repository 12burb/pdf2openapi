from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pdf2openapi import llm
from pdf2openapi.extract import extract


class FakeProvider:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_enrich_merges_and_keeps_citations(vendor_pdf):
    result = extract(vendor_pdf, ocr=False)
    reply = (
        "```json\n"
        + json.dumps(
            {
                "summary": "Better summary",
                "parameters": [
                    {"name": "Idempotency-Key", "in": "header", "schema": {"type": "string"}},
                    {"name": "bad"},
                ],
                "responses": {"201": {"description": "Created"}},
            }
        )
        + "\n```"
    )
    provider = FakeProvider(reply)
    asyncio.run(llm.enrich(result, provider))

    get = result.spec["paths"]["/v2/orders/{order_id}"]["get"]
    assert get["summary"] == "Better summary"
    assert get["x-source-pages"] == [3]
    assert get["x-refined-by"] == "fake"
    names = {(p["in"], p["name"]) for p in get["parameters"]}
    assert names == {("header", "Idempotency-Key"), ("path", "order_id")}
    assert any("Retrieve an order" in p or "order_id" in p for p in provider.prompts)


def test_enrich_failure_keeps_rule_based(vendor_pdf):
    result = extract(vendor_pdf, ocr=False)
    before = result.spec["paths"]["/v2/orders"]["post"]["summary"]
    asyncio.run(llm.enrich(result, FakeProvider("not json")))
    assert result.spec["paths"]["/v2/orders"]["post"]["summary"] == before
    assert any("AI refinement failed" in w for w in result.warnings)


def test_resolve_provider_from_env():
    assert llm.configured_provider_name({}) is None
    assert llm.resolve_provider(env={"OPENAI_API_KEY": "x"}).name == "openai"
    assert llm.resolve_provider(env={"GEMINI_API_KEY": "x"}).name == "gemini"
    p = llm.resolve_provider("ollama", env={})
    assert (p.name, p.model) == ("ollama", "llama3.1")
    with pytest.raises(llm.ProviderError):
        llm.resolve_provider(env={})
    with pytest.raises(llm.ProviderError):
        llm.resolve_provider("gemini", env={})


def _capture(response_body: dict):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_body)

    return seen, httpx.MockTransport(handler)


def test_openai_request_shape():
    seen, transport = _capture({"choices": [{"message": {"content": "{}"}}]})
    p = llm.resolve_provider(env={"OPENAI_API_KEY": "sk-test"}, transport=transport)
    assert asyncio.run(p.complete("sys", "hi")) == "{}"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer sk-test"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_gemini_request_shape():
    seen, transport = _capture({"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
    env = {"GEMINI_API_KEY": "g-test", "PDF2OPENAPI_MODEL": "gemini-x"}
    p = llm.resolve_provider(env=env, transport=transport)
    assert asyncio.run(p.complete("sys", "hi")) == "{}"
    assert seen["url"].endswith("/models/gemini-x:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "g-test"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_provider_http_error_is_reported():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, text="bad key"))
    p = llm.resolve_provider(env={"OPENAI_API_KEY": "x"}, transport=transport)
    with pytest.raises(llm.ProviderError, match="HTTP 401"):
        asyncio.run(p.complete("s", "p"))
