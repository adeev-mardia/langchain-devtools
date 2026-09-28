"""A LangChain Tool that looks up a PyPI package's metadata."""

from __future__ import annotations

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .http_client import HTTPClient, HTTPClientError, get_default_client

PYPI_API_URL = "https://pypi.org/pypi/{package}/json"


class PyPIPackageInput(BaseModel):
    """Input schema for PyPIPackageInfoTool."""

    package: str = Field(
        ...,
        description="The exact PyPI package/distribution name, e.g. 'langchain-core' or 'requests'.",
    )

    @field_validator("package")
    @classmethod
    def _validate_package(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("package must not be empty.")
        if "/" in v or " " in v:
            raise ValueError(f"package must be a single package name: got {v!r}.")
        return v


class PyPIPackageInfoTool(BaseTool):
    """Look up metadata for a package published on the Python Package Index (PyPI).

    Wraps the public, unauthenticated PyPI JSON API
    (`GET https://pypi.org/pypi/{package}/json`), which needs no API key.
    Useful for an agent that wants to check a package's latest version,
    summary, license, homepage, or whether it exists at all before
    recommending it or writing an install command.
    """

    name: str = "pypi_package_info"
    description: str = (
        "Look up a package on PyPI (the Python Package Index) by its exact name. "
        "Returns the latest version, one-line summary, license, author, and "
        "project URLs (homepage/repository, if listed). Use this to check whether "
        "a Python package exists, what its current version is, or what license it "
        "uses, e.g. 'what's the latest version of requests on PyPI?'."
    )
    args_schema: Type[BaseModel] = PyPIPackageInput

    _client: HTTPClient = PrivateAttr()

    def __init__(self, client: Optional[HTTPClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = client or get_default_client()

    def _run(
        self,
        package: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        url = PYPI_API_URL.format(package=package)
        try:
            response = self._client.get(url)
            data = response.json()
        except HTTPClientError as exc:
            if exc.is_not_found:
                return f"No PyPI package named '{package}' was found. Check the spelling of the package name."
            if exc.is_rate_limited:
                return "PyPI rate limit reached. Try again later."
            return f"Error looking up PyPI package '{package}': {exc}"

        info = data.get("info", {})
        project_urls = info.get("project_urls") or {}
        homepage = (
            info.get("home_page")
            or project_urls.get("Homepage")
            or project_urls.get("Repository")
            or "not listed"
        )
        summary = info.get("summary") or "No summary provided."
        license_name = info.get("license") or "not specified"
        # Some packages stuff a full license text into this field; keep it short.
        if len(license_name) > 60:
            license_name = license_name[:57] + "..."

        return (
            f"{info.get('name', package)} {info.get('version', 'unknown version')}: {summary}\n"
            f"Author: {info.get('author') or 'unknown'} | License: {license_name} | "
            f"Requires Python: {info.get('requires_python') or 'not specified'}\n"
            f"Homepage/Repository: {homepage}\n"
            f"PyPI page: https://pypi.org/project/{info.get('name', package)}/"
        )

    async def _arun(
        self,
        package: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(package)
