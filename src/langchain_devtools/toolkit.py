"""Bundles all langchain-devtools tools into a single toolkit."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import BaseTool

from .github_tool import GitHubRepoLookupTool
from .health_check_tool import HTTPHealthCheckTool
from .http_client import HTTPClient, get_default_client
from .pypi_tool import PyPIPackageInfoTool


class DevToolsToolkit:
    """A LangChain-toolkit-style bundle of developer-focused tools.

    Groups :class:`GitHubRepoLookupTool`, :class:`PyPIPackageInfoTool`, and
    :class:`HTTPHealthCheckTool` behind a single `get_tools()` call so an
    agent can be wired up with one line instead of importing and
    instantiating each tool individually:

        from langchain_devtools import DevToolsToolkit

        tools = DevToolsToolkit().get_tools()
        agent = create_react_agent(llm, tools)

    All three tools share one `HTTPClient` (connection pooling + retry
    config) unless a different one is passed in.
    """

    def __init__(self, client: Optional[HTTPClient] = None) -> None:
        self.client = client or get_default_client()

    def get_tools(self) -> List[BaseTool]:
        """Return the list of tool instances in this toolkit."""
        return [
            GitHubRepoLookupTool(client=self.client),
            PyPIPackageInfoTool(client=self.client),
            HTTPHealthCheckTool(client=self.client),
        ]
