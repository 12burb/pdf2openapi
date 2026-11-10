"""Turn vendor PDF API docs into OpenAPI specs."""

from importlib.metadata import PackageNotFoundError, version

from .extract import ExtractResult, extract

__all__ = ["ExtractResult", "extract"]
try:
    __version__ = version("pdf2openapi")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
