"""A LangChain Tool that looks up a public GitHub repository's metadata."""

from __future__ import annotations

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .http_client import HTTPClient, HTTPClientError, get_default_client

GITHUB_API_URL = "https://api.github.com/repos/{owner}/{repo}"


def _clean_ident(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if "/" in value or " " in value:
        raise ValueError(
            f"{field_name} must be a single path segment (no '/' or spaces): got {value!r}."
        )
    return value


class GitHubRepoInput(BaseModel):
    """Input schema for GitHubRepoLookupTool."""

    owner: str = Field(
        ...,
        description="The GitHub username or organization that owns the repository, e.g. 'langchain-ai'.",
    )
    repo: str = Field(
        ...,
        description="The repository name, e.g. 'langchain'.",
    )

    @field_validator("owner")
    @classmethod
    def _validate_owner(cls, v: str) -> str:
        return _clean_ident(v, "owner")

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, v: str) -> str:
        return _clean_ident(v, "repo")


class GitHubRepoLookupTool(BaseTool):
    """Look up metadata for a public GitHub repository.

    Wraps the unauthenticated `GET /repos/{owner}/{repo}` endpoint of the
    GitHub REST API (no API key required, though unauthenticated requests
    are rate-limited to 60/hour per IP). Useful for an agent that needs to
    quickly check a project's popularity, primary language, license, open
    issue count, or last-updated date before recommending or discussing it.
    """

    name: str = "github_repo_lookup"
    description: str = (
        "Look up a public GitHub repository by owner and repo name. "
        "Returns star count, fork count, primary language, license, open issue "
        "count, description, and last-pushed date. Use this when the user asks "
        "about a specific GitHub project's popularity, activity, or metadata, "
        "e.g. 'how many stars does langchain-ai/langchain have?'. "
        "Input must be the owner and repo name separately, not a full URL."
    )
    args_schema: Type[BaseModel] = GitHubRepoInput

    _client: HTTPClient = PrivateAttr()

    def __init__(self, client: Optional[HTTPClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = client or get_default_client()

    def _run(
        self,
        owner: str,
        repo: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        url = GITHUB_API_URL.format(owner=owner, repo=repo)
        try:
            response = self._client.get(url)
            data = response.json()
        except HTTPClientError as exc:
            if exc.is_not_found:
                return f"No GitHub repository found at {owner}/{repo}. Check the owner and repo names."
            if exc.is_rate_limited:
                return (
                    "GitHub API rate limit reached for unauthenticated requests. "
                    "Try again later, or use an authenticated request for a higher limit."
                )
            return f"Error looking up GitHub repo {owner}/{repo}: {exc}"

        license_info = data.get("license") or {}
        return (
            f"{data.get('full_name', f'{owner}/{repo}')}: "
            f"{data.get('description') or 'No description provided.'}\n"
            f"Stars: {data.get('stargazers_count', 'unknown')} | "
            f"Forks: {data.get('forks_count', 'unknown')} | "
            f"Open issues: {data.get('open_issues_count', 'unknown')} | "
            f"Language: {data.get('language') or 'unknown'} | "
            f"License: {license_info.get('spdx_id') or 'none'}\n"
            f"Default branch: {data.get('default_branch', 'unknown')} | "
            f"Last pushed: {data.get('pushed_at', 'unknown')}\n"
            f"URL: {data.get('html_url', url)}"
        )

    async def _arun(
        self,
        owner: str,
        repo: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # No native async HTTP client is used here; delegate to the sync path.
        return self._run(owner, repo)
