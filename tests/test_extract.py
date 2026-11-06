from __future__ import annotations

import pymupdf
import pytest
from conftest import PdfBuilder, build_vendor_pdf

from pdf2openapi import extract, pdf
from pdf2openapi.extract import _clean_summary, _heading_depth, normalize_path
from pdf2openapi.pdf import ocr_available
from pdf2openapi.schema import flatten, parse_type
from pdf2openapi.validate import validate_spec


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/v2/orders/:order_id", "/v2/orders/{order_id}"),
        ("/v2/orders/<int:id>.", "/v2/orders/{id}"),
        ("/v2/orders/{{id}}/items/", "/v2/orders/{id}/items"),
        ("/search?q=x", "/search"),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("int64", {"type": "integer", "format": "int64"}),
        ("string[]", {"type": "array", "items": {"type": "string"}}),
        ("array of strings", {"type": "array", "items": {"type": "string"}}),
        ("Description", None),
    ],
)
def test_parse_type(label, expected):
    assert parse_type(label) == expected


def test_extracts_endpoints_with_citations(vendor_pdf):
    result = extract(vendor_pdf, ocr=False)
    spec = result.spec

    assert spec["info"]["title"] == "Acme Payments API Reference"
    assert spec["info"]["version"] == "2.1"
    assert spec["servers"] == [{"url": "https://api.acme.test"}]
    assert spec["components"]["securitySchemes"] == {
        "apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-Acme-Key"}
    }
    assert set(spec["paths"]) == {"/v2/orders", "/v2/orders/{order_id}"}

    create = spec["paths"]["/v2/orders"]["post"]
    assert create["summary"] == "Create an order"
    assert create["x-source-pages"] == [2]
    body = create["requestBody"]["content"]["application/json"]["schema"]
    assert flatten(body) == {
        "amount": {"type": "integer", "required": True},
        "currency": {"type": "string", "required": True},
        "customer": {"type": "object", "required": False},
        "customer.email": {"type": "string", "required": False},
    }
    assert set(create["responses"]) == {"201", "400"}
    created = create["responses"]["201"]["content"]["application/json"]
    assert created["example"]["id"] == "ord_123"
    assert set(created["schema"]["properties"]) == {"id", "amount", "status"}

    get = spec["paths"]["/v2/orders/{order_id}"]["get"]
    assert get["x-source-pages"] == [3]
    params = {(p["in"], p["name"]): p for p in get["parameters"]}
    assert params[("path", "order_id")]["required"] is True
    assert "required" not in params[("query", "expand")]
    assert set(get["responses"]) == {"200", "404"}


def test_output_is_valid_openapi(vendor_pdf):
    errors, _ = validate_spec(extract(vendor_pdf, ocr=False).spec)
    assert errors == []


def test_unique_operation_ids(vendor_pdf_v2):
    spec = extract(vendor_pdf_v2, ocr=False).spec
    ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()]
    assert len(ids) == len(set(ids))
    assert "postOrders" in ids


@pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
def test_ocr_finds_endpoint_in_screenshot(tmp_path):
    pdf_path = build_vendor_pdf(tmp_path / "shot.pdf", screenshot=True)
    spec = extract(pdf_path).spec
    op = spec["paths"]["/v2/orders/{order_id}"]["delete"]
    assert op["x-source-pages"] == [4]
    assert "204" in op["responses"]


def test_no_endpoints_warns(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((60, 60), "Just a brochure with no API in it.")
    doc.save(tmp_path / "x.pdf")
    result = extract(tmp_path / "x.pdf", ocr=False)
    assert result.spec["paths"] == {}
    assert any("no endpoints found" in w for w in result.warnings)


def test_real_world_layouts(tmp_path):
    """Patterns seen in real vendor PDFs: label rows, reordered columns, wrapped paths."""
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("2.2. SendSms")
    b.row("Method", "URL")
    b.line("POST   https://api.vendor.test/v1/SendSMS")
    b.line("2.2.1   Parameters")
    b.row("Parameter", "Mandatory", "Type", "Description")
    b.row("toNumber", "Yes", "String", "Phone to send to")
    b.row("text", "No", "String", "Message")
    b.line("2.2.2   Responses")
    b.row("Name", "Type", "Description")
    b.row("status", "string", "Result")
    b.line("Lookup a number")
    b.line("GET /v1/AssignNumber?Number=xxx&format=json")
    b.line("Download media")
    b.line("GET /v1/media/{file.ext}/")
    b.line("content")
    path = b.save(tmp_path / "real.pdf")

    spec = extract(path, ocr=False).spec
    send = spec["paths"]["/v1/SendSMS"]["post"]
    assert send["summary"] == "SendSms"
    assert "parameters" not in send
    body = send["requestBody"]["content"]["application/json"]["schema"]
    assert flatten(body) == {
        "toNumber": {"type": "string", "required": True},
        "text": {"type": "string", "required": False},
    }
    resp = send["responses"]["200"]["content"]["application/json"]["schema"]
    assert list(resp["properties"]) == ["status"]

    lookup = spec["paths"]["/v1/AssignNumber"]["get"]
    assert [p["name"] for p in lookup["parameters"]] == ["Number", "format"]
    assert "/v1/media/{file_ext}/content" in spec["paths"]
    assert validate_spec(spec)[0] == []


def test_heading_after_trailing_slash_is_not_joined(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("List orders")
    b.line("GET /orders/")
    b.line("Response")
    b.line("200 OK")
    b.line("Wrapped")
    b.line("GET /v1/accounts/{account")
    b.line("_id}/balance")
    path = b.save(tmp_path / "slash.pdf")
    paths = extract(path, ocr=False).spec["paths"]
    assert "/orders" in paths
    assert "/v1/accounts/{account_id}/balance" in paths
    assert not any("Response" in p for p in paths)


def test_method_and_url_on_separate_lines(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("Get PAR URL")
    b.line("- Method: GET")
    b.line("- Path: /objectstore/getParURL")
    b.row("Parameter", "Value Type", "Description")
    b.row("filename", "String", "File name")
    b.line("Payment request")
    b.line("API URL")
    b.line("https://tt.pay.test/plugins/payments/request")
    b.line("HTTP")
    b.line("POST")
    b.row("Name", "Type", "Required")
    b.row("total", "number", "Yes")
    path = b.save(tmp_path / "split.pdf")
    spec = extract(path, ocr=False).spec
    assert [p["name"] for p in spec["paths"]["/objectstore/getParURL"]["get"]["parameters"]] == [
        "filename"
    ]
    post = spec["paths"]["/plugins/payments/request"]["post"]
    assert "total" in post["requestBody"]["content"]["application/json"]["schema"]["properties"]
    assert {"url": "https://tt.pay.test"} in spec["servers"]


def test_uri_without_method_is_guessed(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("4.1.1 All Active Bids Report")
    b.line("4.1.1.2.1 URI to access the Web API")
    b.line("/api-AllActiveBidsReport/v1")
    b.line("4.1.1.3.1 API Request Parameters")
    b.row("Name", "Type", "Required")
    b.row("gasDay", "date", "Yes")
    path = b.save(tmp_path / "guess.pdf")
    result = extract(path, ocr=False)
    op = result.spec["paths"]["/api-AllActiveBidsReport/v1"]["post"]
    assert op["x-method-guessed"] is True
    assert any("guessed" in w for w in result.warnings)


def test_xsd_table_rows_become_body_fields(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("4.1.1 All Active Bids Report")
    b.line("URI to access the Web API")
    b.line("/api-AllActiveBidsReport/v1")
    b.line("API Request Parameters")
    b.row("xml", "json")
    b.row("1", "MOS", "xs:string", "Yes")
    b.row("1", "GAS_DAY", "xs:date", "No")
    path = b.save(tmp_path / "xsd.pdf")
    op = extract(path, ocr=False).spec["paths"]["/api-AllActiveBidsReport/v1"]["post"]
    body = op["requestBody"]["content"]["application/json"]["schema"]
    assert body["properties"]["GAS_DAY"]["format"] == "date"
    assert set(body["properties"]) == {"MOS", "GAS_DAY"}
    assert body["required"] == ["MOS"]
    assert op["summary"] == "All Active Bids Report"


def _ops(spec):
    return sorted(f"{m.upper()} {p}" for p, item in spec["paths"].items() for m in item)


def test_split_endpoint_next_to_complete_endpoint(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("GET /a")
    b.line("Method: POST")
    b.line("Path: /b")
    spec = extract(b.save(tmp_path / "mixed.pdf"), ocr=False).spec
    assert _ops(spec) == ["GET /a", "POST /b"]


def test_adjacent_split_endpoints_keep_their_own_paths(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("Method: GET")
    b.line("Path: /a")
    b.line("Method: POST")
    b.line("Path: /b")
    b.line("Path: /c")
    b.line("Method: PUT")
    spec = extract(b.save(tmp_path / "adjacent.pdf"), ocr=False).spec
    assert _ops(spec) == ["GET /a", "POST /b", "PUT /c"]


def test_long_section_keeps_later_pages(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("2.1 Create order")
    b.line("POST /orders")
    for _ in range(3):
        b.new_page()
        b.line("Notes about orders")
    b.line("201 Created")
    b.new_page()
    b.line("3 Appendix")
    b.line("404 Not Found")
    spec = extract(b.save(tmp_path / "long.pdf"), ocr=False).spec
    assert set(spec["paths"]["/orders"]["post"]["responses"]) == {"201"}


def test_mixed_server_prefix_paths(tmp_path):
    from conftest import PdfBuilder

    b = PdfBuilder().new_page()
    b.line("Base URL: https://api.test/v2")
    b.line("GET /v2/orders")
    b.line("GET /health")
    spec = extract(b.save(tmp_path / "prefix.pdf"), ocr=False).spec
    assert spec["servers"] == [{"url": "https://api.test/v2"}]
    assert _ops(spec) == ["GET /health", "GET /orders"]


def test_path_wrapped_inside_table_cell(tmp_path):
    b = PdfBuilder().new_page()
    b.line("Permissions", 14)
    b.row("API", "Action")
    page, y = b.page, b.y
    page.insert_text((60, y), "POST /v1.0/", fontsize=10)
    page.insert_text((190, y), "vod:asset:create", fontsize=10)
    for k, part in enumerate(["{project_id}/asset/", "template-", "collection"], 1):
        page.insert_text((60, y + 13 * k), part, fontsize=10)
    b.y = y + 80
    b.line("Create a collection", 14)
    b.line("POST /v1.0/{project_id}/asset/template-collection")
    b.line("Creates a template collection.")
    b.line("200 OK")
    result = extract(b.save(tmp_path / "wrapped.pdf"), ocr=False)

    assert list(result.spec["paths"]) == ["/v1.0/{project_id}/asset/template-collection"]
    op = result.spec["paths"]["/v1.0/{project_id}/asset/template-collection"]["post"]
    assert op["summary"] == "Create a collection"


def test_running_headers_are_dropped(tmp_path):
    b = PdfBuilder()
    for n, (title, path) in enumerate(
        [("List orders", "/orders"), ("List refunds", "/refunds"), ("List users", "/users")], 1
    ):
        b.new_page()
        b.line(f"Acme API Reference   Page {n} of 3")
        b.line(title, 14)
        b.line(f"GET {path}")
        b.line(f"Returns all {path[1:]}.")
        b.line("200 OK")
    result = extract(b.save(tmp_path / "headers.pdf"), ocr=False)

    assert len(result.spec["paths"]) == 3
    assert not any("Acme API Reference" in text for text in result.sections.values())
    assert result.spec["paths"]["/refunds"]["get"]["summary"] == "List refunds"


def test_generic_numbered_headings_do_not_become_the_section(tmp_path):
    b = PdfBuilder().new_page()
    b.line("4.1 Bids API", 14)
    b.line("4.1.1 Overview")
    b.line("Returns bids.")
    b.line("4.1.2 URI to access the Web API")
    b.line("POST /api-bids/v1")
    b.line("4.1.3 Specifications")
    b.line("Body Parameters")
    b.row("Name", "Type", "Required", "Description")
    b.row("bid_id", "string", "Yes", "Bid ID")
    b.line("4.2 Offers API", 14)
    b.line("POST /api-offers/v1")
    b.line("200 OK")
    spec = extract(b.save(tmp_path / "generic.pdf"), ocr=False).spec

    bids = spec["paths"]["/api-bids/v1"]["post"]
    assert bids["summary"] == "Bids API"
    assert "bid_id" in flatten(bids["requestBody"]["content"]["application/json"]["schema"])


@pytest.mark.parametrize("text", ["vod:asset:create", "vod:category:li \u221a", "}", "\u25cf"])
def test_permission_names_and_symbols_are_not_summaries(text):
    assert _clean_summary(text) == ""


def test_zero_row_number_is_not_a_heading():
    assert _heading_depth("0   Top level hierarchy for query elements") is None
    assert (
        _heading_depth("4.1.2 Bid Information Shipper View \u2013 Normal Pricing Strategy API") == 3
    )


def _rpc_pdf(tmp_path):
    b = PdfBuilder().new_page()
    b.line("SYNO.Acme.List", 14)
    b.line("Method")
    b.line("list")
    b.line("Description:")
    b.line("Enumerate files in a given folder.")
    b.line("Request:")
    b.row("Parameter", "Description", "Value", "Availability")
    b.row("folder_path", "String", "(None)", "1 and later")
    b.row("offset", "files skipped", "Integer", "0")
    b.line("Example:")
    b.line("GET /webapi/entry.cgi?api=SYNO.Acme.List&version=2&method=list&")
    b.line("folder_path=%22%2Fvideo%22")
    b.line("Response:")
    b.row("Parameter", "Type", "Description", "Availability")
    b.row("total", "Integer", "Total number of files.", "1 and later")
    b.line("Method")
    b.line("stop")
    b.line("Description:")
    b.line("Stop the listing task.")
    b.line("Request:")
    b.row("Parameter", "Type", "Description", "Availability")
    b.row("taskid", "String", "A unique task ID.", "1 and later")
    b.line("Example:")
    b.line("GET /webapi/entry.cgi?api=SYNO.Acme.List&version=1&method%3Dstop&taskid=%22x%22")
    return extract(b.save(tmp_path / "rpc.pdf"), ocr=False)


def test_query_rpc_methods_become_separate_operations(tmp_path):
    result = _rpc_pdf(tmp_path)
    paths = result.spec["paths"]
    assert sorted(paths) == [
        "/webapi/entry.cgi?api=SYNO.Acme.List&method=list",
        "/webapi/entry.cgi?api=SYNO.Acme.List&method=stop",
    ]
    op = paths["/webapi/entry.cgi?api=SYNO.Acme.List&method=list"]["get"]
    assert op["summary"] == "Enumerate files in a given folder."
    params = {p["name"]: p for p in op["parameters"]}
    assert params["method"]["schema"]["enum"] == ["list"] and params["method"]["required"]
    assert params["api"]["schema"]["enum"] == ["SYNO.Acme.List"]
    assert params["offset"]["schema"]["type"] == "integer"
    assert {"folder_path", "version"} <= set(params)
    assert "total" in op["responses"]["200"]["content"]["application/json"]["schema"]["properties"]
    stop = paths["/webapi/entry.cgi?api=SYNO.Acme.List&method=stop"]["get"]
    assert {p["name"] for p in stop["parameters"]} >= {"taskid", "version"}
    assert "folder_path" not in {p["name"] for p in stop["parameters"]}
    assert validate_spec(result.spec)[0] == []


def test_ordinary_query_parameters_stay_parameters(tmp_path):
    b = PdfBuilder().new_page()
    b.line("Search orders", 14)
    b.line("GET /v1/orders?status=open&limit=10")
    result = extract(b.save(tmp_path / "q.pdf"), ocr=False)
    assert list(result.spec["paths"]) == ["/v1/orders"]
    names = {p["name"] for p in result.spec["paths"]["/v1/orders"]["get"]["parameters"]}
    assert names == {"status", "limit"}


@pytest.mark.parametrize("text", ["Description:", "Response:", "Example:"])
def test_label_lines_are_not_running_headers(tmp_path, text):
    b = PdfBuilder()
    for path in ["/a", "/b", "/c"]:
        b.new_page()
        b.line(f"GET {path}")
        b.line(text)
        b.line(f"Returns the {path[1:]} record and its owner.")
        b.line("The record is cached for one minute.")
        b.line("See the error codes chapter for failures.")
    result = extract(b.save(tmp_path / "labels.pdf"), ocr=False)
    assert all(text in body for body in result.sections.values())


def test_wrapped_prose_does_not_switch_table_context(tmp_path):
    b = PdfBuilder().new_page()
    b.line("Get statistics", 14)
    b.line("GET /v1/stats")
    b.line("Request header parameters")
    b.row("Parameter", "Mandatory", "Type", "Description")
    b.row("X-Token", "No", "String", "Token. The value of X-")
    b.line("Subject-Token in the response")
    b.line("header is the user token.")
    b.row("X-Sdk-Date", "No", "String", "Time when a request is sent.")
    result = extract(b.save(tmp_path / "wrap.pdf"), ocr=False)
    params = {p["name"]: p["in"] for p in result.spec["paths"]["/v1/stats"]["get"]["parameters"]}
    assert params.get("X-Sdk-Date") == "header"


@pytest.mark.parametrize("cs,alpha", [(pymupdf.csRGB, True), (pymupdf.csCMYK, False)])
def test_ocr_converts_rgba_and_cmyk_images(cs, alpha):
    pix = pymupdf.Pixmap(cs, pymupdf.IRect(0, 0, 8, 8), alpha)
    pix.clear_with(255)
    png = pdf._png_for_ocr(pix)
    out = pymupdf.Pixmap(png)
    assert (out.n, out.alpha) == (3, 0)


def test_ai_hint_flags_thin_operations():
    from pdf2openapi.extract import ai_hint

    ok = {"summary": "List orders", "responses": {}}
    spec = {"paths": {"/orders": {"get": ok}}}
    assert ai_hint(spec) is None
    spec["paths"]["/orders"]["post"] = {"summary": "Create order"}
    spec["paths"]["/refunds"] = {"get": {}}
    hint = ai_hint(spec)
    assert hint.startswith("2 of 3 endpoints look incomplete")
    assert "POST /orders" in hint and "GET /refunds" in hint
    assert ai_hint({"paths": {}}).startswith("no endpoints found")


def test_multipart_example_becomes_form_body(tmp_path):
    b = PdfBuilder().new_page()
    b.line("Upload", 14)
    b.line("Example:")
    b.line("POST /webapi/entry.cgi")
    b.line("Content-type: multipart/form-data, boundary=AaB03x")
    b.line("--AaB03x")
    b.line('content-disposition: form-data; name="api"')
    b.line("SYNO.FileStation.Upload")
    b.line("--AaB03x")
    b.line('content-disposition: form-data; name="method"')
    b.line("upload")
    b.line("--AaB03x")
    b.line('content-disposition: form-data; name="version"')
    b.line("2")
    b.line("--AaB03x")
    b.line('content-disposition: form-data; name="overwrite"')
    b.line("true")
    b.line("--AaB03x")
    b.line('content-disposition: form-data; name="file"; filename="file1.txt"')
    b.line("Content-Type: application/octet-stream")
    b.line("--AaB03x--")
    op = extract(b.save(tmp_path / "upload.pdf"), ocr=False).spec["paths"]["/webapi/entry.cgi"]
    op = op["post"]
    assert op["summary"] == "SYNO.FileStation.Upload upload"
    form = op["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert form["properties"]["api"] == {"type": "string", "enum": ["SYNO.FileStation.Upload"]}
    assert form["properties"]["version"]["type"] == "integer"
    assert form["properties"]["overwrite"]["type"] == "boolean"
    assert form["properties"]["file"] == {"type": "string", "format": "binary"}
    assert set(form["required"]) == {"api", "method", "file"}
