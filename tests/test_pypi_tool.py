"""Tests for PyPIPackageInfoTool. All HTTP is mocked via `responses`."""

from __future__ import annotations

import pytest
import responses
from pydantic import ValidationError

from langchain_devtools.pypi_tool import PyPIPackageInfoTool, PyPIPackageInput

SAMPLE_PACKAGE_JSON = {
    "info": {
        "name": "requests",
        "version": "2.32.0",
        "summary": "Python HTTP for Humans.",
        "author": "Kenneth Reitz",
        "license": "Apache-2.0",
        "requires_python": ">=3.8",
        "home_page": "https://requests.readthedocs.io",
        "project_urls": {"Source": "https://github.com/psf/requests"},
    }
}


@responses.activate
def test_run_success_returns_readable_summary():
    responses.add(
        responses.GET,
        "https://pypi.org/pypi/requests/json",
        json=SAMPLE_PACKAGE_JSON,
        status=200,
    )
    tool = PyPIPackageInfoTool()
    result = tool.run({"package": "requests"})
    assert "requests 2.32.0" in result
    assert "Apache-2.0" in result
    assert "Kenneth Reitz" in result


@responses.activate
def test_run_404_returns_clean_not_found_string():
    responses.add(
        responses.GET,
        "https://pypi.org/pypi/this-package-does-not-exist-xyz/json",
        status=404,
    )
    tool = PyPIPackageInfoTool()
    result = tool.run({"package": "this-package-does-not-exist-xyz"})
    assert "No PyPI package named" in result
    assert "Traceback" not in result


@responses.activate
def test_run_network_error_returns_clean_string(monkeypatch):
    import requests

    tool = PyPIPackageInfoTool()

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(tool._client.session, "request", raise_timeout)
    result = tool.run({"package": "requests"})
    assert "Error looking up PyPI package" in result
    assert "Traceback" not in result


@responses.activate
def test_run_handles_missing_optional_fields_gracefully():
    responses.add(
        responses.GET,
        "https://pypi.org/pypi/bare-package/json",
        json={"info": {"name": "bare-package", "version": "0.1.0"}},
        status=200,
    )
    tool = PyPIPackageInfoTool()
    result = tool.run({"package": "bare-package"})
    assert "bare-package 0.1.0" in result
    assert "not listed" in result or "unknown" in result


def test_input_schema_rejects_empty_package():
    with pytest.raises(ValidationError):
        PyPIPackageInput(package="")


def test_input_schema_rejects_package_with_slash():
    with pytest.raises(ValidationError):
        PyPIPackageInput(package="foo/bar")


def test_input_schema_strips_whitespace():
    parsed = PyPIPackageInput(package="  requests  ")
    assert parsed.package == "requests"
