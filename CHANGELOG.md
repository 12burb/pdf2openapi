# Changelog

## 0.1.0 - unreleased

First release.

### Extract
- Convert vendor PDF API docs to OpenAPI 3.0 with no AI and no API key: rule-based parsing of
  method/path lines, parameter tables, JSON examples and authentication sections.
- OCR of screenshots and scanned pages with Tesseract (optional).
- Every operation cites its source pages in `x-source-pages`.
- Query-RPC APIs (`/entry.cgi?api=X&method=Y`, as in Synology's docs) become one operation per call.
- Endpoints whose method is not stated are guessed and marked `x-method-guessed`.
- Raw `multipart/form-data` examples become a multipart request body with typed fields and a
  binary `file` field.
- A `tip:` line names endpoints that came out incomplete, judged without AI.

### Optional AI refinement
- Bring your own model: OpenAI or any OpenAI-compatible API, Gemini, or local Ollama.
- AI output is merged into the rule-based draft; required parameters are never dropped and
  invalid parameter locations are corrected.
- A bad key or unknown model stops refinement after one call with one clear warning.

### Diff and validate
- `diff` compares PDFs, YAML/JSON specs or URLs and flags breaking changes (removed endpoints,
  new required inputs, type changes, auth and OAuth scope changes, server changes).
  `--fail-on-change` / `--fail-on-breaking` for CI.
- `validate` checks the spec and can probe GET/HEAD endpoints against a sandbox.
- URL fetching refuses private addresses (including via redirects and DNS) unless
  `PDF2OPENAPI_ALLOW_PRIVATE_URLS=1`.

### MCP
- `pdf2openapi mcp` runs a stdio MCP server with `extract_openapi`, `read_pdf_pages`,
  `diff_api_docs` and `validate_openapi`. With `use_ai=true` and no key, it uses the client's
  own model through MCP sampling.
