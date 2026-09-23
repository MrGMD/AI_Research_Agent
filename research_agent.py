import os

# -------------------------------------------------------------------
# CrewAI + Groq compatibility patch
#
# CrewAI 1.15.x can add a `cache_breakpoint` field to messages.
# Groq rejects that field with a 400 Bad Request.
#
# Disable that injection before the agent executor sends messages.
# This is a known CrewAI/Groq compatibility issue.
# -------------------------------------------------------------------
try:
    import crewai.llms.cache as _crewai_cache

    def _disable_cache_breakpoint(message):
        return message

    _crewai_cache.mark_cache_breakpoint = _disable_cache_breakpoint

    # Some CrewAI versions import mark_cache_breakpoint directly
    # into these executor modules, so patch those references too.
    try:
        import crewai.experimental.agent_executor as _agent_executor
        _agent_executor.mark_cache_breakpoint = _disable_cache_breakpoint
    except Exception:
        pass

    try:
        import crewai.agents.crew_agent_executor as _crew_agent_executor
        _crew_agent_executor.mark_cache_breakpoint = _disable_cache_breakpoint
    except Exception:
        pass

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

        # Groq may expose executed browser-search results on the
        # message object. Add them when available.
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

    # CrewAI uses LiteLLM for the Groq provider.
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=12000,
    )

    # Single CrewAI agent.
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
