# langchain-devtools

A small toolkit of custom [LangChain](https://python.langchain.com/) `Tool`s that wrap real, free,
**no-API-key-required** public HTTP APIs a developer would actually want an agent to use:

| Tool | Wraps | What it returns |
|---|---|---|
| `github_repo_lookup` | `GET api.github.com/repos/{owner}/{repo}` (unauthenticated) | stars, forks, language, license, open issues, last-pushed date |
| `pypi_package_info` | `GET pypi.org/pypi/{package}/json` (public) | latest version, summary, license, author, project URLs |
| `http_health_check` | GET/HEAD to any URL | status code, latency, up/down verdict |

## Why this isn't a toy

Every "custom LangChain tool" tutorial online wraps a fake weather API or a hardcoded dictionary. These three
tools hit **real, live, production APIs** that require no signup, no key, and no billing:

- The GitHub REST API's unauthenticated tier (60 req/hour/IP) is exactly what a small agent or CI script uses.
- PyPI's JSON API is the same one `pip` itself and dependency-scanning tools rely on.
- The health-check tool is a general-purpose primitive — "is this thing up" — useful for almost any
  infra/dev-facing agent.

Because they hit real APIs, they also have to handle **real failure modes**: 404s for typo'd names, GitHub's
rate limiting, DNS/connection failures, timeouts, and malformed responses — all without ever leaking a raw
Python traceback back into an LLM's context window. That error-handling is the actual engineering content of
this repo, not boilerplate around it.

## Design

- **`http_client.py`** — a single shared `HTTPClient` (a thin `requests.Session` wrapper) used by all three
  tools. It centralizes timeouts, retry/backoff (via `urllib3.Retry`, retrying on 502/503/504 but *not* on
  404/4xx, which are legitimate results), and translates every failure mode into one exception type,
  `HTTPClientError`, carrying `status_code`, `is_not_found`, and `is_rate_limited` flags. Each tool catches
  that one exception type and turns it into a short, LLM-readable string — never a stack trace.
- **Pydantic input schemas** — each tool has an `args_schema` (`GitHubRepoInput`, `PyPIPackageInput`,
  `HealthCheckInput`) with field validators that reject empty strings, whitespace-only input, owner/repo
  values containing `/` or spaces, and URLs missing a scheme — so the LLM gets a clear validation error
  instead of a confusing downstream 404.
- **Rich `description`s** — since LangChain tools describe *themselves* to the LLM that's deciding whether and
  how to call them, each tool's docstring/description spells out exactly what it does, what it returns, and an
  example of when to use it.
- **`toolkit.py`** — `DevToolsToolkit().get_tools()` returns all three tools sharing one `HTTPClient` instance
  (connection pooling, one retry config), LangChain-toolkit style, so wiring an agent up is one line.

## Install

```bash
pip install -e ".[dev]"
```

(Requires Python 3.9+. Runtime deps: `langchain-core`, `pydantic>=2`, `requests`. Dev deps: `pytest`, `responses`.)

## Usage

```python
from langchain_devtools import DevToolsToolkit

tools = DevToolsToolkit().get_tools()

# Call a tool directly:
github_tool = tools[0]
print(github_tool.run({"owner": "langchain-ai", "repo": "langchain"}))
# -> "langchain-ai/langchain: Build context-aware reasoning applications ...
#     Stars: 90000 | Forks: 14000 | Open issues: 250 | Language: Python | License: MIT
#     ..."
```

### Plugging into a real LangChain agent

```python
from langchain_devtools import DevToolsToolkit
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI  # or any other chat model

llm = ChatOpenAI(model="gpt-4o-mini")
tools = DevToolsToolkit().get_tools()

agent = create_react_agent(llm, tools)
result = agent.invoke({
    "messages": [("user", "How many stars does langchain-ai/langchain have, "
                            "and is pypi.org up right now?")]
})
print(result["messages"][-1].content)
```

See [`scripts/demo.py`](scripts/demo.py) for a fuller, runnable-once-you-add-a-key example. It is **not** part
of the test suite and is never executed with a real API key in this repo — this environment has no LLM
provider key configured, and the script is meant to be read as a wiring example.

## Individual tools

- **`GitHubRepoLookupTool`** (`github_repo_lookup`) — input: `owner`, `repo`. Rejects empty or slash-containing
  values.
- **`PyPIPackageInfoTool`** (`pypi_package_info`) — input: `package`. Rejects empty or slash-containing values.
- **`HTTPHealthCheckTool`** (`http_health_check`) — input: `url` (must include `http(s)://` and a host),
  optional `method` (`HEAD` default, or `GET`).

## Tests

```bash
python3 -m pytest -v
```

The test suite is **fully offline and deterministic** — no network calls are made and no API keys are needed
to develop or test this package. Every HTTP call is mocked with the [`responses`](https://github.com/getsentry/responses)
library (and, for connection/timeout failures, `unittest.mock`-style monkeypatching of the underlying
`requests.Session.request`). Coverage includes, for each tool:

- a successful response, parsed into a clean summary string
- a 404 / not-found response, turned into a clean "not found" message
- a simulated network error (`ConnectionError`/`Timeout`), turned into a clean error message — never a raw
  traceback
- GitHub-style rate limiting (429 and GitHub's 403-with-rate-limit-message)
- Pydantic input validation (empty/whitespace input, malformed values) at the schema level
- the toolkit correctly bundling all three tools and sharing one `HTTPClient`

## License

MIT — see [LICENSE](LICENSE).
