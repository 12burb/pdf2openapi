"""Optional AI refinement. Users bring their own model: OpenAI, Gemini, Ollama, or an MCP client."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from .extract import ExtractResult

SYSTEM_PROMPT = (
    "You convert excerpts of vendor API documentation into OpenAPI 3.0 operation objects. "
    "Use only facts stated in the excerpt; never invent fields. "
    "Reply with one JSON object and nothing else."
)
MAX_EXCERPT_CHARS = 12000
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash", "ollama": "llama3.1"}
ALLOWED_KEYS = ("summary", "description", "parameters", "requestBody", "responses")


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    async def complete(self, system: str, prompt: str) -> str: ...


class _HTTPProvider:
    name = "http"

    def __init__(self, model: str, transport: httpx.AsyncBaseTransport | None = None):
        self.model = model
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=180, transport=self._transport)

    async def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        async with self._client() as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name} returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()


class OpenAIProvider(_HTTPProvider):
    """OpenAI or any OpenAI-compatible endpoint (set OPENAI_BASE_URL)."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None = None, **kw: Any):
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    async def complete(self, system: str, prompt: str) -> str:
        data = await self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return data["choices"][0]["message"]["content"]


class GeminiProvider(_HTTPProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, **kw: Any):
        super().__init__(model, **kw)
        self.api_key = api_key

    async def complete(self, system: str, prompt: str) -> str:
        data = await self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
            {"x-goog-api-key": self.api_key},
        )
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class OllamaProvider(_HTTPProvider):
    """Local models via Ollama. No key, nothing leaves the machine."""

    name = "ollama"

    def __init__(self, model: str, host: str | None = None, **kw: Any):
        super().__init__(model, **kw)
        self.host = (host or "http://localhost:11434").rstrip("/")

    async def complete(self, system: str, prompt: str) -> str:
        data = await self._post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            {},
        )
        return data["message"]["content"]


class CallableProvider:
    """Wraps any async (system, prompt) -> text function, e.g. MCP sampling."""

    def __init__(self, name: str, fn: Callable[[str, str], Awaitable[str]]):
        self.name = name
        self._fn = fn

    async def complete(self, system: str, prompt: str) -> str:
        return await self._fn(system, prompt)


def configured_provider_name(env: dict[str, str] | None = None) -> str | None:
    env = dict(os.environ) if env is None else env
    explicit = env.get("PDF2OPENAPI_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    return None


def resolve_provider(
    name: str | None = None,
    model: str | None = None,
    env: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Provider:
    env = dict(os.environ) if env is None else env
    name = (name or configured_provider_name(env) or "").lower()
    if not name:
        raise ProviderError(
            "no AI provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY, "
            "or use --provider ollama for a free local model."
        )
    model = model or env.get("PDF2OPENAPI_MODEL") or DEFAULT_MODELS.get(name)
    if name == "openai":
        key = env.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return OpenAIProvider(key, model, env.get("OPENAI_BASE_URL"), transport=transport)
    if name == "gemini":
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        return GeminiProvider(key, model, transport=transport)
    if name == "ollama":
        return OllamaProvider(model, env.get("OLLAMA_HOST"), transport=transport)
    raise ProviderError(f"unknown provider '{name}' (use openai, gemini or ollama)")


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model reply contained no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model reply was not a JSON object")
    return value


def _build_prompt(key: str, op: dict[str, Any], excerpt: str) -> str:
    draft = {k: v for k, v in op.items() if k in ALLOWED_KEYS}
    return (
        f"Endpoint: {key}\n"
        f"Source pages: {op.get('x-source-pages')}\n\n"
        "Draft operation from a rule-based parser (may be incomplete or wrong):\n"
        f"{json.dumps(draft, indent=2)}\n\n"
        "Documentation excerpt:\n<<<\n"
        f"{excerpt[:MAX_EXCERPT_CHARS]}\n>>>\n\n"
        "Return a corrected OpenAPI 3.0 operation object with only these keys: "
        "summary, description, parameters, requestBody, responses. "
        "Each parameter needs name, in (path|query|header|cookie), required and schema."
    )


def _clean_operation(reply: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("summary", "description"):
        if isinstance(reply.get(key), str) and reply[key].strip():
            out[key] = reply[key].strip()
    params = reply.get("parameters")
    if isinstance(params, list):
        good = [
            p
            for p in params
            if isinstance(p, dict)
            and isinstance(p.get("name"), str)
            and p.get("in") in ("path", "query", "header", "cookie")
        ]
        for p in good:
            p.setdefault("schema", {"type": "string"})
            if p["in"] == "path":
                p["required"] = True
        out["parameters"] = good
    body = reply.get("requestBody")
    if isinstance(body, dict) and isinstance(body.get("content"), dict):
        out["requestBody"] = body
    responses = reply.get("responses")
    if isinstance(responses, dict) and responses:
        cleaned = {}
        for code, resp in responses.items():
            if isinstance(resp, dict):
                resp.setdefault("description", "Response")
                cleaned[str(code)] = resp
        if cleaned:
            out["responses"] = cleaned
    return out


async def enrich(result: ExtractResult, provider: Provider, concurrency: int = 4) -> ExtractResult:
    """Ask the model to correct each operation from its source excerpt, in place."""
    sem = asyncio.Semaphore(concurrency)

    async def refine(path: str, method: str, op: dict[str, Any]) -> None:
        key = f"{method.upper()} {path}"
        excerpt = result.sections.get(key)
        if not excerpt:
            return
        async with sem:
            try:
                reply = await provider.complete(SYSTEM_PROMPT, _build_prompt(key, op, excerpt))
                cleaned = _clean_operation(parse_json_object(reply))
            except Exception as exc:  # noqa: BLE001 - keep the rule-based result on any failure
                result.warnings.append(
                    f"{key}: AI refinement failed ({exc}); kept rule-based result"
                )
                return
        path_params = set(re.findall(r"\{(\w+)\}", path))
        if "parameters" in cleaned:
            declared = {p["name"] for p in cleaned["parameters"] if p["in"] == "path"}
            for name in sorted(path_params - declared):
                cleaned["parameters"].append(
                    {"name": name, "in": "path", "required": True, "schema": {"type": "string"}}
                )
        op.update(cleaned)
        if "responses" in cleaned:
            op.pop("x-inferred-responses", None)
        op["x-refined-by"] = provider.name

    tasks = [
        refine(path, method, op)
        for path, item in result.spec.get("paths", {}).items()
        for method, op in item.items()
    ]
    await asyncio.gather(*tasks)
    return result
