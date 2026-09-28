"""Example: wiring langchain-devtools into a LangChain agent.

This script is illustrative, not executable as-is: it requires an LLM
provider API key (e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY) which is not
available in this environment, and it is intentionally NOT part of the test
suite. Read it to see how the tools plug into a real agent.

To actually run it, install an LLM integration (e.g. `langchain-openai` or
`langchain-anthropic`), set the relevant API key in your environment, and
uncomment the `agent.invoke(...)` call at the bottom.
"""

from __future__ import annotations

from langchain_devtools import DevToolsToolkit


def build_agent():
    """Build a ReAct-style agent wired up with the DevToolsToolkit tools."""
    # Swap in whichever chat model you have credentials for. Examples:
    #
    #   from langchain_openai import ChatOpenAI
    #   llm = ChatOpenAI(model="gpt-4o-mini")
    #
    #   from langchain_anthropic import ChatAnthropic
    #   llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    #
    # We don't import either here so this file has no LLM-provider
    # dependency at import time.
    from langchain.chat_models.base import BaseChatModel  # type: ignore

    llm: "BaseChatModel"  # placeholder type hint; assign a real model above.

    tools = DevToolsToolkit().get_tools()

    # `create_react_agent` (from `langgraph.prebuilt`) builds a small graph
    # that lets the LLM call any of `tools` in a loop until it has an answer.
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(llm, tools)
    return agent


def main() -> None:
    print("Tools in the toolkit:")
    for t in DevToolsToolkit().get_tools():
        print(f"  - {t.name}: {t.description.splitlines()[0]}")

    print(
        "\nThis demo does not call an LLM (no API key is configured in this "
        "environment). To run it for real:\n"
        "  1. pip install langchain langgraph langchain-openai  # or another provider\n"
        "  2. export OPENAI_API_KEY=...\n"
        "  3. Uncomment the agent-building code in build_agent() and the\n"
        "     agent.invoke(...) call below, then run this script again.\n"
    )

    # agent = build_agent()
    # result = agent.invoke(
    #     {
    #         "messages": [
    #             (
    #                 "user",
    #                 "How many stars does langchain-ai/langchain have on GitHub, "
    #                 "and what's the latest version of langchain-core on PyPI?",
    #             )
    #         ]
    #     }
    # )
    # print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
