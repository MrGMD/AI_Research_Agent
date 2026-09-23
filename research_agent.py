import os


# ---------------------------------------------------------------------------
# CrewAI + Groq compatibility fix
# ---------------------------------------------------------------------------
#
# CrewAI 1.15.x can add `cache_breakpoint` to messages for its prompt-cache
# handling. Groq's API rejects that property on messages:
#
#   property 'cache_breakpoint' is unsupported
#
# This is a known CrewAI issue affecting non-Anthropic providers such as
# Groq. We disable the marker at the CrewAI level AND remove it at the final
# LiteLLM boundary as a safety net.
#
# The second patch is intentionally included because some CrewAI versions
# import the cache function into an executor module before execution.
# ---------------------------------------------------------------------------

try:
    import crewai.llms.cache as _crewai_cache

    def _no_cache_breakpoint(message):
        return message

    _crewai_cache.mark_cache_breakpoint = _no_cache_breakpoint

    # CrewAI's older executor.
    try:
        import crewai.agents.crew_agent_executor as _crew_agent_executor

        _crew_agent_executor.mark_cache_breakpoint = _no_cache_breakpoint
    except Exception:
        pass

    # CrewAI's experimental/current executor.
    try:
        import crewai.experimental.agent_executor as _agent_executor

        _agent_executor.mark_cache_breakpoint = _no_cache_breakpoint
    except Exception:
        pass

except Exception:
    pass


# Final safety net:
# remove cache_breakpoint immediately before LiteLLM sends the request.
try:
    import litellm

    _original_litellm_completion = litellm.completion

    def _completion_without_groq_cache_breakpoint(*args, **kwargs):
        messages = kwargs.get("messages")

        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    message.pop("cache_breakpoint", None)

                    content = message.get("content")

                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                block.pop("cache_breakpoint", None)

        # Disable LiteLLM-side caching for this application.
        kwargs["caching"] = False

        return _original_litellm_completion(*args, **kwargs)

    litellm.completion = _completion_without_groq_cache_breakpoint

except Exception:
    pass


from groq import Groq
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool


class GroqWebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The research question or search query to investigate.",
    )


class GroqWebSearchTool(BaseTool):
    name: str = "Groq Web Research"
    description: str = (
        "Search the live web using Groq's built-in browser search. "
        "Use this tool whenever current, factual, or source-based "
        "information is needed."
    )
    args_schema: type[BaseModel] = GroqWebSearchInput

    def _run(self, query: str) -> str:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a web research assistant. Search the live web "
                        "and return factual findings with source information. "
                        "Prefer reliable, primary, academic, government, "
                        "and reputable news sources when appropriate."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[{"type": "browser_search"}],
            tool_choice="required",
            temperature=0.2,
            max_completion_tokens=8000,
        )

        message = response.choices[0].message
        content = message.content or ""

        # Groq can return executed browser-search results on the message.
        executed_tools = getattr(message, "executed_tools", None)

        if executed_tools:
            content += "\n\n## Web Search Sources\n"

            for tool_result in executed_tools:
                search_results = getattr(
                    tool_result,
                    "search_results",
                    None,
                )

                if not search_results:
                    continue

                results = getattr(
                    search_results,
                    "results",
                    None,
                )

                if not results:
                    continue

                for result in results:
                    title = getattr(result, "title", "")
                    url = getattr(result, "url", "")
                    snippet = getattr(result, "content", "")

                    if title or url or snippet:
                        content += (
                            f"- {title}\n"
                            f"  URL: {url}\n"
                            f"  Summary: {snippet}\n"
                        )

        return content


def run_research(topic: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    # CrewAI routes this through LiteLLM.
    #
    # IMPORTANT:
    # `groq/` is the LiteLLM provider prefix.
    # `openai/gpt-oss-120b` is the actual model ID on Groq.
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=12000,
    )

    # Exactly ONE CrewAI agent.
    researcher = Agent(
        role="Senior AI Research Analyst",
        goal=(
            "Research the user's topic thoroughly using reliable web "
            "sources and produce an accurate, well-structured research report."
        ),
        backstory=(
            "You are an experienced research analyst. You gather information "
            "from multiple sources, compare evidence, identify important "
            "findings, and clearly separate facts from interpretation. "
            "You never invent sources or unsupported claims."
        ),
        tools=[GroqWebSearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
    )

    research_task = Task(
        description=f"""
Research the following topic:

"{topic}"

You MUST use the Groq Web Research tool to search the live web
before preparing the report.

Requirements:
1. Use multiple relevant sources.
2. Prefer reliable and authoritative sources.
3. Prefer recent information when the topic requires it.
4. Identify the most important facts and findings.
5. Compare sources where useful.
6. Do not make unsupported claims.
7. Distinguish factual information from interpretation.
8. Include important dates, statistics, organizations,
   people, or developments when relevant.
9. Never invent a source or URL.
10. Include source names and URLs available from the research tool.

Use this structure:

# Research Report

## Executive Summary

## Introduction

## Background

## Key Findings

## Detailed Analysis

## Current Developments

## Challenges and Limitations

## Conclusion

## Sources

Topic:
{topic}
""",
        expected_output=(
            "A detailed research report based on live web research, "
            "with an executive summary, introduction, background, "
            "key findings, detailed analysis, current developments, "
            "challenges and limitations, conclusion, and sources."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    result = crew.kickoff()

    return str(result)
