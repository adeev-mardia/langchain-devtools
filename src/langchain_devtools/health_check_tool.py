"""A LangChain Tool that checks whether an HTTP endpoint is up."""

from __future__ import annotations

from typing import Optional, Type
from urllib.parse import urlparse

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .http_client import HTTPClient, HTTPClientError, get_default_client


class HealthCheckInput(BaseModel):
    """Input schema for HTTPHealthCheckTool."""

    url: str = Field(
        ...,
        description="The full URL to check, including scheme, e.g. 'https://example.com'.",
    )
    method: str = Field(
        default="HEAD",
        description="HTTP method to use: 'HEAD' (default, lighter) or 'GET'.",
    )

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty.")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"url must start with http:// or https:// : got {v!r}."
            )
        if not parsed.netloc:
            raise ValueError(f"url is missing a host: got {v!r}.")
        return v

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ("HEAD", "GET"):
            raise ValueError(f"method must be 'HEAD' or 'GET': got {v!r}.")
        return v


class HTTPHealthCheckTool(BaseTool):
    """Check whether an arbitrary HTTP/HTTPS endpoint is up and responsive.

    Performs a lightweight GET or HEAD request against a given URL and
    reports the resulting status code, round-trip latency, and a simple
    up/down verdict. Useful for an agent troubleshooting "is this site/API
    down for everyone" style questions, or verifying a URL before
    recommending it.

    Note: some servers reject HEAD requests (405) even when they are
    healthy; if that happens, try again with method='GET'.
    """

    name: str = "http_health_check"
    description: str = (
        "Check if a given URL is up by making an HTTP request to it. Returns the "
        "HTTP status code, response latency in milliseconds, and whether the "
        "endpoint appears to be up. Use this when asked things like 'is "
        "example.com down?' or 'check if this API endpoint is responding'. "
        "Defaults to a HEAD request; pass method='GET' if HEAD isn't supported."
    )
    args_schema: Type[BaseModel] = HealthCheckInput

    _client: HTTPClient = PrivateAttr()

    def __init__(self, client: Optional[HTTPClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = client or get_default_client()

    def _run(
        self,
        url: str,
        method: str = "HEAD",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            response = self._client.request(method, url, raise_for_status=False)
        except HTTPClientError as exc:
            return f"{url} appears to be DOWN: {exc}"

        latency_ms = round(response.elapsed_seconds * 1000, 1)
        status = response.status_code
        if status == 429:
            verdict = "responding but rate limiting requests"
        elif status < 400:
            verdict = "UP"
        elif status == 404:
            verdict = "UP (but returned 404 Not Found for this path)"
        else:
            verdict = "responding but returned an error status"

        return (
            f"{url} is {verdict}. HTTP status: {status}. Latency: {latency_ms} ms "
            f"(method: {method})."
        )

    async def _arun(
        self,
        url: str,
        method: str = "HEAD",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(url, method)
