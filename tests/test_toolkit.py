"""Tests for DevToolsToolkit."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from langchain_devtools import DevToolsToolkit
from langchain_devtools.github_tool import GitHubRepoLookupTool
from langchain_devtools.health_check_tool import HTTPHealthCheckTool
from langchain_devtools.pypi_tool import PyPIPackageInfoTool


def test_get_tools_returns_three_tools():
    tools = DevToolsToolkit().get_tools()
    assert len(tools) == 3
    for t in tools:
        assert isinstance(t, BaseTool)


def test_get_tools_includes_expected_tool_types():
    tools = DevToolsToolkit().get_tools()
    types = {type(t) for t in tools}
    assert types == {GitHubRepoLookupTool, PyPIPackageInfoTool, HTTPHealthCheckTool}


def test_tools_share_the_same_http_client():
    toolkit = DevToolsToolkit()
    tools = toolkit.get_tools()
    for t in tools:
        assert t._client is toolkit.client


def test_all_tools_have_unique_names_and_descriptions():
    tools = DevToolsToolkit().get_tools()
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    for t in tools:
        assert t.description and len(t.description) > 20
