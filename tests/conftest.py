from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

COLUMNS = (60, 190, 290, 370)


def _text_image(lines: list[str]) -> bytes:
    """Render lines of text into a PNG, like a screenshot pasted into a PDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=40 + 30 * len(lines))
    for i, line in enumerate(lines):
        page.insert_text((20, 40 + 30 * i), line, fontsize=16)
    png = page.get_pixmap(dpi=200).tobytes("png")
    doc.close()
    return png


class PdfBuilder:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = None
        self.y = 0.0

    def new_page(self) -> PdfBuilder:
        self.page = self.doc.new_page()
        self.y = 60
        return self

    def line(self, text: str, size: float = 10) -> PdfBuilder:
        self.page.insert_text((60, self.y), text, fontsize=size)
        self.y += size + 8
        return self

    def row(self, *cells: str) -> PdfBuilder:
        for x, cell in zip(COLUMNS, cells, strict=False):
            self.page.insert_text((x, self.y), cell, fontsize=10)
        self.y += 18
        return self

    def json(self, value: object) -> PdfBuilder:
        for text in json.dumps(value, indent=2).splitlines():
            self.page.insert_text((70, self.y), text, fontsize=9, fontname="cour")
            self.y += 13
        self.y += 6
        return self

    def screenshot(self, lines: list[str]) -> PdfBuilder:
        rect = pymupdf.Rect(60, self.y, 460, self.y + 40 + 24 * len(lines))
        self.page.insert_image(rect, stream=_text_image(lines))
        self.y = rect.y1 + 10
        return self

    def save(self, path: Path) -> Path:
        self.doc.set_metadata({"title": "Acme Payments API Reference"})
        self.doc.save(path)
        self.doc.close()
        return path


def build_vendor_pdf(path: Path, version: int = 1, screenshot: bool = False) -> Path:
    b = PdfBuilder().new_page()
    b.line("Acme Payments API Reference", 18)
    b.line(f"Version 2.{version}")
    b.line("Base URL: https://api.acme.test/v2")
    b.line("Authentication")
    b.line("Send your secret key in the X-Acme-Key header with every request.")
    b.line("Endpoints")
    b.line("POST /v2/orders ........ 2")
    b.line("GET /v2/orders/:order_id ........ 3")

    b.new_page()
    b.line("Create an order", 14)
    b.line("POST /v2/orders")
    b.line("Creates a new order for a customer.")
    b.line("Body Parameters")
    b.row("Name", "Type", "Required", "Description")
    b.row("amount", "integer" if version == 1 else "number", "Yes", "Amount in cents")
    b.row("currency", "string", "Yes", "ISO 4217 code")
    b.row("customer.email", "string", "No", "Customer email")
    if version == 2:
        b.row("reference", "string", "Yes", "Your order reference")
    b.line("Response")
    b.line("201 Created")
    b.json({"id": "ord_123", "amount": 500, "status": "pending"})
    b.line("400 Bad Request")

    b.new_page()
    b.line("Retrieve an order", 14)
    b.line("GET /v2/orders/:order_id")
    b.line("Path Parameters")
    b.row("order_id", "string", "Yes", "The order ID")
    b.line("Query Parameters")
    b.row("expand", "string", "No", "Related objects to include")
    b.line("200 OK")
    b.line("404 Not Found")

    if version == 2:
        b.new_page()
        b.line("List refunds", 14)
        b.line("GET /v2/refunds")
        b.line("Query Parameters")
        b.row("limit", "integer", "No", "Max results")
        b.line("200 OK")

    if screenshot:
        b.new_page()
        b.line("Cancel an order", 14)
        b.screenshot(["DELETE /v2/orders/{order_id}", "204 No Content"])
    return b.save(path)


@pytest.fixture
def vendor_pdf(tmp_path: Path) -> Path:
    return build_vendor_pdf(tmp_path / "acme-v1.pdf", version=1)


@pytest.fixture
def vendor_pdf_v2(tmp_path: Path) -> Path:
    return build_vendor_pdf(tmp_path / "acme-v2.pdf", version=2)
