"""Tests for GitHubRepoLookupTool. All HTTP is mocked via `responses`."""

from __future__ import annotations

import pytest
import responses
from pydantic import ValidationError

from langchain_devtools.github_tool import GitHubRepoInput, GitHubRepoLookupTool

SAMPLE_REPO_JSON = {
    "full_name": "langchain-ai/langchain",
    "description": "Build context-aware reasoning applications",
    "stargazers_count": 90000,
    "forks_count": 14000,
    "open_issues_count": 250,
    "language": "Python",
    "license": {"spdx_id": "MIT"},
    "default_branch": "master",
    "pushed_at": "2026-09-20T00:00:00Z",
    "html_url": "https://github.com/langchain-ai/langchain",
}


@responses.activate
def test_run_success_returns_readable_summary():
    responses.add(
        responses.GET,
        "https://api.github.com/repos/langchain-ai/langchain",
        json=SAMPLE_REPO_JSON,
        status=200,
    )
    tool = GitHubRepoLookupTool()
    result = tool.run({"owner": "langchain-ai", "repo": "langchain"})
    assert isinstance(result, str)
    assert "langchain-ai/langchain" in result
    assert "90000" in result
    assert "MIT" in result


@responses.activate
def test_run_404_returns_clean_not_found_string():
    responses.add(
        responses.GET,
        "https://api.github.com/repos/nope/doesnotexist",
        status=404,
    )
    tool = GitHubRepoLookupTool()
    result = tool.run({"owner": "nope", "repo": "doesnotexist"})
    assert isinstance(result, str)
    assert "No GitHub repository found" in result
    # must not leak a raw traceback / exception repr
    assert "Traceback" not in result


@responses.activate
def test_run_network_error_returns_clean_string(monkeypatch):
    import requests

    tool = GitHubRepoLookupTool()

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(tool._client.session, "request", raise_connection_error)
    result = tool.run({"owner": "langchain-ai", "repo": "langchain"})
    assert isinstance(result, str)
    assert "Error looking up GitHub repo" in result
    assert "Traceback" not in result


@responses.activate
def test_run_rate_limited_returns_clean_string():
    responses.add(
        responses.GET,
        "https://api.github.com/repos/a/b",
        json={"message": "API rate limit exceeded"},
        status=403,
    )
    tool = GitHubRepoLookupTool()
    result = tool.run({"owner": "a", "repo": "b"})
    assert "rate limit" in result.lower()


def test_input_schema_rejects_empty_owner():
    with pytest.raises(ValidationError):
        GitHubRepoInput(owner="", repo="langchain")


def test_input_schema_rejects_empty_repo():
    with pytest.raises(ValidationError):
        GitHubRepoInput(owner="langchain-ai", repo="")


def test_input_schema_rejects_owner_with_slash():
    with pytest.raises(ValidationError):
        GitHubRepoInput(owner="langchain-ai/langchain", repo="langchain")


def test_input_schema_strips_whitespace():
    parsed = GitHubRepoInput(owner="  langchain-ai  ", repo=" langchain ")
    assert parsed.owner == "langchain-ai"
    assert parsed.repo == "langchain"


@responses.activate
def test_tool_run_with_empty_owner_returns_clean_error_not_exception():
    tool = GitHubRepoLookupTool()
    # BaseTool.run validates args against args_schema and raises a
    # (non-network) exception for bad input; assert it's not a raw crash
    # with no useful message.
    with pytest.raises(Exception) as exc_info:
        tool.run({"owner": "", "repo": "langchain"})
    assert "empty" in str(exc_info.value).lower()
