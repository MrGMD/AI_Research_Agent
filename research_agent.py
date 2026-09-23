import os

from groq import Groq
from crewai import Agent, Crew, Task, BaseLLM


# ============================================================================
# CrewAI + Groq compatibility fix
# ============================================================================
#
# CrewAI can add `cache_breakpoint` to messages.
# Groq does not accept that property.
# We remove it before sending the request.
# ============================================================================

try:
    import crewai.llms.cache as _crewai_cache

    def _no_cache_breakpoint(message):
        return message

    _crewai_cache.mark_cache_breakpoint = _no_cache_breakpoint

    try:
        import crewai.agents.crew_agent_executor as _crew_agent_executor

        _crew_agent_executor.mark_cache_breakpoint = _no_cache_breakpoint
    except Exception:
        pass

    try:
        import crewai.experimental.agent_executor as _agent_executor

        _agent_executor.mark_cache_breakpoint = _no_cache_breakpoint
    except Exception:
        pass

except Exception:
    pass


# ============================================================================
# Custom Groq LLM for CrewAI
# ============================================================================
#
# This keeps CrewAI as the agent framework but sends the actual LLM request
# directly to Groq.
#
# IMPORTANT:
# - ONE CrewAI agent
# - ONE Groq API request from our application
# - GPT-OSS 120B
# - Native Groq browser_search
# - Low reasoning effort
# - No separate web-search model
# - No Serper
# - No Tavily
# - No second API key
# ============================================================================

class GroqNativeLLM(BaseLLM):

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_completion_tokens: int = 3500,
    ):

        super().__init__(
            model=model,
            temperature=temperature,
        )

        self.api_key = api_key
        self.model = model
        self.max_completion_tokens = max_completion_tokens

        self.client = Groq(
            api_key=api_key
        )

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs,
    ):

        # ---------------------------------------------------------------
        # Convert CrewAI messages into normal Groq messages.
        # ---------------------------------------------------------------

        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        clean_messages = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get(
                "role",
                "user",
            )

            content = message.get(
                "content",
                "",
            )

            # Remove CrewAI-only cache information.
            message_copy = {
                "role": role,
                "content": content,
            }

            clean_messages.append(
                message_copy
            )

        # ---------------------------------------------------------------
        # Groq native browser search.
        #
        # Groq performs the search/tool loop server-side.
        # ---------------------------------------------------------------

        response = self.client.chat.completions.create(

            model=self.model,

            messages=clean_messages,

            tools=[
                {
                    "type": "browser_search"
                }
            ],

            # Force the agent to perform web research.
            tool_choice="required",

            # Low reasoning = fewer reasoning tokens.
            reasoning_effort="low",

            # Do not return the internal reasoning field.
            include_reasoning=False,

            temperature=self.temperature,

            # Keep the final answer controlled.
            max_completion_tokens=self.max_completion_tokens,
        )

        message = response.choices[0].message

        content = message.content or ""

        # ---------------------------------------------------------------
        # Add source information returned by Groq.
        # ---------------------------------------------------------------

        executed_tools = getattr(
            message,
            "executed_tools",
            None,
        )

        if executed_tools:

            source_lines = []

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

                    title = getattr(
                        result,
                        "title",
                        "",
                    )

                    url = getattr(
                        result,
                        "url",
                        "",
                    )

                    if title and url:

                        source_lines.append(
                            f"- [{title}]({url})"
                        )

            if source_lines:

                content += (
                    "\n\n## Sources\n\n"
                    + "\n".join(
                        source_lines
                    )
                )

        return content

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 131072


# ============================================================================
# Main Research Function
# ============================================================================

def run_research(topic: str) -> str:

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    # ------------------------------------------------------------------------
    # ONE GPT-OSS 120B LLM
    #
    # Reduced output size to control token usage.
    # ------------------------------------------------------------------------

    llm = GroqNativeLLM(

        api_key=api_key,

        model="openai/gpt-oss-120b",

        temperature=0.2,

        max_completion_tokens=3500,
    )

    # ------------------------------------------------------------------------
    # EXACTLY ONE CREWAI AGENT
    #
    # No custom web-search tool.
    # Groq itself provides browser search.
    # ------------------------------------------------------------------------

    researcher = Agent(

        role="Senior AI Research Analyst",

        goal=(
            "Research the user's topic using reliable current web sources "
            "and produce an accurate, concise research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You investigate topics using reliable web sources, "
            "compare important evidence, and clearly distinguish "
            "facts from interpretation. "
            "You never invent sources or unsupported claims."
        ),

        llm=llm,

        # No local tools.
        tools=[],

        verbose=False,

        allow_delegation=False,

        cache=False,
    )

    # ------------------------------------------------------------------------
    # SINGLE RESEARCH TASK
    # ------------------------------------------------------------------------

    research_task = Task(

        description=f"""
Research this topic:

"{topic}"

IMPORTANT:

Use the built-in web search available through the Groq model.

Keep the research focused and token-efficient.

Use only the most relevant sources needed to answer the topic.

Prefer:
- official sources
- academic sources
- government sources
- reputable organizations
- reputable news sources

Do not perform unnecessary searches.

Do not repeat the same information.

Produce a concise but useful research report.

Structure:

# Research Report

## Executive Summary

Give a short summary of the most important findings.

## Introduction

Briefly introduce the topic.

## Key Findings

List the most important factual findings.

## Detailed Analysis

Explain the findings clearly.

## Current Developments

Include recent developments when relevant.

## Limitations

Mention important limitations or uncertainty.

## Conclusion

Give a concise conclusion.

## Sources

List the important sources used.

Do not invent sources or URLs.

Topic:

{topic}
""",

        expected_output=(
            "A concise research report based on live web research, "
            "including an executive summary, introduction, key findings, "
            "analysis, current developments, limitations, conclusion, "
            "and sources."
        ),

        agent=researcher,
    )

    # ------------------------------------------------------------------------
    # ONE CREW
    # ------------------------------------------------------------------------

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            research_task
        ],

        verbose=False,
    )

    # ------------------------------------------------------------------------
    # EXECUTE
    # ------------------------------------------------------------------------

    result = crew.kickoff()

    return str(result)
