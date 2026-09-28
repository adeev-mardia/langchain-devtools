"""Tests for the shared HTTPClient. All HTTP is mocked via `responses`."""

from __future__ import annotations

import requests
import responses

from langchain_devtools.http_client import HTTPClient, HTTPClientError


@responses.activate
def test_get_success_returns_response():
    responses.add(
        responses.GET,
        "https://example.com/ok",
        json={"hello": "world"},
        status=200,
    )
    client = HTTPClient()
    resp = client.get("https://example.com/ok")
    assert resp.status_code == 200
    assert resp.json() == {"hello": "world"}


@responses.activate
def test_get_404_raises_not_found_by_default():
    responses.add(responses.GET, "https://example.com/missing", status=404)
    client = HTTPClient()
    try:
        client.get("https://example.com/missing")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert exc.is_not_found is True
        assert exc.status_code == 404


@responses.activate
def test_get_404_allowed_returns_response_when_allow_404():
    responses.add(responses.GET, "https://example.com/missing", status=404)
    client = HTTPClient()
    resp = client.get("https://example.com/missing", allow_404=True)
    assert resp.status_code == 404


@responses.activate
def test_get_429_raises_rate_limited():
    responses.add(responses.GET, "https://example.com/limited", status=429)
    client = HTTPClient(max_retries=0)
    try:
        client.get("https://example.com/limited")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert exc.is_rate_limited is True


@responses.activate
def test_github_style_403_rate_limit_body_detected():
    responses.add(
        responses.GET,
        "https://api.github.com/repos/x/y",
        json={"message": "API rate limit exceeded for x.x.x.x"},
        status=403,
    )
    client = HTTPClient(max_retries=0)
    try:
        client.get("https://api.github.com/repos/x/y")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert exc.is_rate_limited is True


@responses.activate
def test_other_4xx_raises_plain_error():
    responses.add(responses.GET, "https://example.com/bad", status=400)
    client = HTTPClient(max_retries=0)
    try:
        client.get("https://example.com/bad")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert exc.status_code == 400
        assert exc.is_not_found is False
        assert exc.is_rate_limited is False


def test_connection_error_raises_clean_message(monkeypatch):
    client = HTTPClient()

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(client.session, "request", raise_connection_error)
    try:
        client.get("https://unreachable.example.com")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert "unreachable" in str(exc) or "connect" in str(exc).lower()


def test_timeout_raises_clean_message(monkeypatch):
    client = HTTPClient()

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(client.session, "request", raise_timeout)
    try:
        client.get("https://slow.example.com")
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert "timed out" in str(exc).lower()


@responses.activate
def test_malformed_json_raises_clean_error():
    responses.add(
        responses.GET,
        "https://example.com/not-json",
        body="<html>not json</html>",
        status=200,
        content_type="text/html",
    )
    client = HTTPClient()
    resp = client.get("https://example.com/not-json")
    try:
        resp.json()
        assert False, "expected HTTPClientError"
    except HTTPClientError as exc:
        assert "not valid JSON" in str(exc)
