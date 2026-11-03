"""Turn vendor PDF API docs into OpenAPI specs."""

from .extract import ExtractResult, extract

__all__ = ["ExtractResult", "extract"]
__version__ = "0.1.0"
