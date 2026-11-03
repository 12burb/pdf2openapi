# pdf2openapi

Turn a vendor's PDF API docs into an OpenAPI spec, and get told when the PDF changes.

Some vendors only ship a 60-page PDF: endpoints are buried in prose and screenshots, the
auth examples are wrong, and new versions arrive without a changelog. People end up
retyping it all into Postman and only notice changes when production breaks.
`pdf2openapi` does the retyping for you and diffs each new PDF against the last one.

- **Runs locally.** No account, no server, nothing uploaded. Your PDFs stay on your machine.
- **Works without AI.** A rule-based parser plus OCR handles most well-structured docs.
- **AI is optional, with your own model:** OpenAI, Gemini, a local Ollama model, or the model
  of the MCP client you are already using (Claude, Cursor, and others).
- **Cites its sources.** Every operation has `x-source-pages` so you can check it.

## Install

```bash
pip install git+https://github.com/12burb/pdf2openapi
# optional, to read screenshots and scanned pages:
sudo apt install tesseract-ocr   # macOS: brew install tesseract
```

## Use

```bash
# PDF -> OpenAPI
pdf2openapi extract vendor-api.pdf -o vendor-api.yaml

# What changed between two versions? (PDFs, YAML/JSON specs, or URLs)
pdf2openapi diff vendor-api-v1.pdf vendor-api-v2.pdf

# Is the spec valid? Do the documented GET endpoints actually respond?
pdf2openapi validate vendor-api.yaml --base-url https://sandbox.vendor.com -H "X-Api-Key: $KEY"
```

Import the generated spec into Postman, Insomnia or Bruno, or generate a typed client
with any OpenAPI generator.

Example `diff` output:

```markdown
## API changes (2 breaking)

### New endpoints
- `GET /v2/refunds`

### Changed endpoints
- **BREAKING** `POST /v2/orders`: body field 'amount' type integer -> number
- **BREAKING** `POST /v2/orders`: body field 'reference' added (required)
```

## Optional AI refinement (bring your own model)

The rule-based parser struggles with docs that describe fields in free prose. Add `--ai` and
each endpoint's source pages are sent to a model, which corrects the draft. Only the pages
for that endpoint are sent, and only when you pass `--ai`.

| Provider | Setup | Notes |
| --- | --- | --- |
| OpenAI | `export OPENAI_API_KEY=...` | Any OpenAI-compatible API works via `OPENAI_BASE_URL` |
| Gemini | `export GEMINI_API_KEY=...` | |
| Ollama | `--provider ollama` | Free and local, no key |
| Your MCP client | nothing | See below |

```bash
pdf2openapi extract vendor-api.pdf --ai                     # uses whichever key is set
pdf2openapi extract vendor-api.pdf --provider ollama --model llama3.1
```

`PDF2OPENAPI_PROVIDER` and `PDF2OPENAPI_MODEL` set defaults.

## Use from Claude, Cursor or any MCP client

`pdf2openapi mcp` runs an MCP server over stdio. Add it to your client config, for example
Claude Desktop's `claude_desktop_config.json` or Cursor's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pdf2openapi": { "command": "pdf2openapi", "args": ["mcp"] }
  }
}
```

Tools:

- `extract_openapi(pdf_path, use_ai, output_path, format)`
- `read_pdf_pages(pdf_path, pages)` returns page-marked text, including OCR of screenshots
- `diff_api_docs(old, new, format)`
- `validate_openapi(spec_path)`

With `use_ai=true` and no API key configured, the server asks the client's own model to do
the refinement through MCP sampling, so no extra key is needed. Clients without sampling
support get the rule-based spec plus `read_pdf_pages`, which lets the assistant check the
spec against the source pages itself.

## Get alerted when the vendor PDF changes

Commit the current spec, then run this on a schedule. Example GitHub Actions workflow:

```yaml
name: vendor-api-watch
on:
  schedule: [{ cron: "0 8 * * *" }]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install git+https://github.com/12burb/pdf2openapi
      - run: pdf2openapi diff specs/vendor.yaml https://vendor.example.com/api-docs.pdf --fail-on-change
```

A failing run means the PDF changed. The log shows what changed and which changes break clients.

URLs must resolve to public addresses (each redirect is checked; at most 5 redirects and
100 MB). To diff a document on a private network, set `PDF2OPENAPI_ALLOW_PRIVATE_URLS=1`.

## Limitations

- Endpoints are found by lines like `POST /v1/orders`, or by a method and URL on separate
  lines (`Method: GET` / `Path: /orders`). When a doc labels a URI but never states the
  method, the method is guessed and the operation is marked `x-method-guessed: true`.
  Docs that only describe endpoints in prose need `--ai`.
- Tables are rebuilt from text positions. Unusual layouts may lose columns. Check the
  `warning:` lines and `x-source-pages`.
- OCR quality depends on screenshot resolution.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check .
```

## License

MIT
