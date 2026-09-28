"""Thin, shared HTTP client used by every tool in this package.

All three tools in ``langchain-devtools`` talk to a plain JSON/HTTP API over
the network. Rather than duplicating timeout handling, retry logic, and
exception-to-clean-error translation in each tool, that logic lives here
once.

The client deliberately raises a single exception type, :class:`HTTPClientError`,
for every failure mode (timeouts, connection errors, non-2xx responses, bad
JSON). Tools catch that one type and turn it into a short, LLM-readable
string instead of leaking a raw traceback back into the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 10.0  # seconds
DEFAULT_USER_AGENT = "langchain-devtools/0.1 (+https://github.com/adeev/langchain-devtools)"


class HTTPClientError(Exception):
    """Raised for any network, timeout, or HTTP-status failure.

    Attributes:
        status_code: The HTTP status code if a response was received, else None.
        is_not_found: True when the failure was a 404.
        is_rate_limited: True when the failure was a 429 (or a GitHub-style
            403 rate-limit response).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        is_not_found: bool = False,
        is_rate_limited: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_not_found = is_not_found
        self.is_rate_limited = is_rate_limited


@dataclass
class HTTPResponse:
    """A minimal, tool-friendly wrapper around a completed HTTP request."""

    status_code: int
    elapsed_seconds: float
    headers: dict
    url: str
    _raw: requests.Response

    def json(self) -> Any:
        try:
            return self._raw.json()
        except ValueError as exc:
            raise HTTPClientError(
                f"Response from {self.url} was not valid JSON."
            ) from exc

    @property
    def text(self) -> str:
        return self._raw.text


class HTTPClient:
    """A small `requests.Session` wrapper with sane defaults for all tools.

    - A bounded connection/read timeout so a hanging server can't stall an
      agent run forever.
    - Automatic retries (with backoff) for transient failures (connection
      resets, 502/503/504) but NOT for 404s or 4xx client errors, which are
      treated as legitimate, immediately-returned results.
    - A single `HTTPClientError` raised for every failure case, carrying
      enough structure (status_code, is_not_found, is_rate_limited) for
      callers to write a good error message without inspecting exception
      internals.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 2,
    ) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        allow_404: bool = False,
        raise_for_status: bool = True,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        """Perform an HTTP request and return a clean response or raise.

        Args:
            method: HTTP verb, e.g. "GET" or "HEAD".
            url: Fully-qualified URL to request.
            params: Optional query parameters.
            headers: Optional extra headers.
            allow_404: If True, a 404 response is returned normally (not
                raised) so the caller can build a "not found" message with
                context. If False, a 404 raises HTTPClientError with
                is_not_found=True. Ignored when raise_for_status=False.
            raise_for_status: If False, any HTTP status (including 4xx/5xx)
                is returned as a normal HTTPResponse instead of raising —
                used by callers (like the health-check tool) that care about
                "did the server respond at all", not just 2xx bodies.
                Connection errors and timeouts still raise regardless.
            timeout: Per-request timeout override.

        Raises:
            HTTPClientError: on timeout or connection failure always; on
                rate limiting, or (unless allow_404) a 404, or any other
                non-2xx status, only when raise_for_status is True.
        """
        try:
            raw = self.session.request(
                method,
                url,
                params=params,
                headers=headers,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise HTTPClientError(f"Request to {url} timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise HTTPClientError(
                f"Could not connect to {url}. The host may be down or unreachable."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise HTTPClientError(f"Request to {url} failed: {exc}") from exc

        response = HTTPResponse(
            status_code=raw.status_code,
            elapsed_seconds=raw.elapsed.total_seconds(),
            headers=dict(raw.headers),
            url=url,
            _raw=raw,
        )

        if not raise_for_status:
            return response

        if raw.status_code == 404 and not allow_404:
            raise HTTPClientError(
                f"Resource not found at {url} (404).",
                status_code=404,
                is_not_found=True,
            )

        if raw.status_code == 429 or (
            raw.status_code == 403 and "rate limit" in raw.text.lower()
        ):
            raise HTTPClientError(
                f"Rate limited by {url} (status {raw.status_code}). Try again later.",
                status_code=raw.status_code,
                is_rate_limited=True,
            )

        if raw.status_code >= 400 and not (allow_404 and raw.status_code == 404):
            raise HTTPClientError(
                f"Request to {url} failed with HTTP {raw.status_code}.",
                status_code=raw.status_code,
            )

        return response

    def get(self, url: str, **kwargs) -> HTTPResponse:
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs) -> HTTPResponse:
        return self.request("HEAD", url, **kwargs)


_default_client: Optional[HTTPClient] = None


def get_default_client() -> HTTPClient:
    """Return a process-wide shared HTTPClient (created lazily)."""
    global _default_client
    if _default_client is None:
        _default_client = HTTPClient()
    return _default_client
