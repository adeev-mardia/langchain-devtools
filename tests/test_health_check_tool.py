"""Tests for HTTPHealthCheckTool. All HTTP is mocked via `responses`."""

from __future__ import annotations

import pytest
import responses
from pydantic import ValidationError

from langchain_devtools.health_check_tool import HealthCheckInput, HTTPHealthCheckTool


@responses.activate
def test_run_success_reports_up():
    responses.add(responses.HEAD, "https://example.com", status=200)
    tool = HTTPHealthCheckTool()
    result = tool.run({"url": "https://example.com"})
    assert "is UP" in result
    assert "200" in result
    assert "ms" in result


@responses.activate
def test_run_404_reports_up_but_not_found():
    responses.add(responses.HEAD, "https://example.com/missing", status=404)
    tool = HTTPHealthCheckTool()
    result = tool.run({"url": "https://example.com/missing"})
    assert "404" in result
    assert "UP" in result


@responses.activate
def test_run_server_error_reports_not_up():
    responses.add(responses.HEAD, "https://example.com/broken", status=503)
    tool = HTTPHealthCheckTool()
    result = tool.run({"url": "https://example.com/broken"})
    assert "responding but returned an error status" in result
    assert "503" in result


@responses.activate
def test_run_get_method():
    responses.add(responses.GET, "https://example.com/api", status=200, json={})
    tool = HTTPHealthCheckTool()
    result = tool.run({"url": "https://example.com/api", "method": "GET"})
    assert "is UP" in result
    assert "method: GET" in result


def test_run_connection_error_reports_down(monkeypatch):
    import requests

    tool = HTTPHealthCheckTool()

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(tool._client.session, "request", raise_connection_error)
    result = tool.run({"url": "https://unreachable.example.com"})
    assert "DOWN" in result
    assert "Traceback" not in result


def test_run_timeout_reports_down(monkeypatch):
    import requests

    tool = HTTPHealthCheckTool()

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(tool._client.session, "request", raise_timeout)
    result = tool.run({"url": "https://slow.example.com"})
    assert "DOWN" in result
    assert "timed out" in result.lower()


def test_input_schema_rejects_empty_url():
    with pytest.raises(ValidationError):
        HealthCheckInput(url="")


def test_input_schema_rejects_url_without_scheme():
    with pytest.raises(ValidationError):
        HealthCheckInput(url="example.com")


def test_input_schema_rejects_bad_method():
    with pytest.raises(ValidationError):
        HealthCheckInput(url="https://example.com", method="DELETE")


def test_input_schema_defaults_to_head():
    parsed = HealthCheckInput(url="https://example.com")
    assert parsed.method == "HEAD"


def test_input_schema_normalizes_method_case():
    parsed = HealthCheckInput(url="https://example.com", method="get")
    assert parsed.method == "GET"
