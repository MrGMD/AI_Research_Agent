import os

from crewai import Agent, Crew, Task, LLM
from crewai_tools import SerperDevTool


def run_research(topic: str) -> str:
    """
    Run the single-agent research workflow.
    """

    # -----------------------------------------
    # 1. Configure Groq LLM
    # -----------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=12000,
    )

    # -----------------------------------------
    # 2. Create web search tool
    # -----------------------------------------

    search_tool = SerperDevTool(
        n_results=10
    )

    # -----------------------------------------
    # 3. Create ONE CrewAI agent
    # -----------------------------------------

    researcher = Agent(
        role="Senior AI Research Analyst",

        goal=(
            "Research the user's topic using reliable and "
            "relevant web sources and produce an accurate, "
            "well-structured research report."
        ),

        backstory=(
            "You are an experienced research analyst who "
            "specializes in finding, analyzing, and organizing "
            "information from multiple sources. You carefully "
            "distinguish facts from opinions and avoid making "
            "unsupported claims."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # -----------------------------------------
    # 4. Create research task
    # -----------------------------------------

    research_task = Task(
        description=f"""
        Conduct thorough research on this topic:

        "{topic}"

        Research requirements:

        1. Find relevant and reliable sources.
        2. Use multiple sources rather than relying on one source.
        3. Prefer recent information when the topic requires it.
        4. Identify the most important facts and findings.
        5. Compare information from different sources when useful.
        6. Avoid unsupported claims.
        7. Clearly distinguish facts, analysis, and opinions.
        8. Include important dates, statistics, organizations,
           people, or developments when relevant.
        9. Do not invent sources or citations.
        10. Provide source names and URLs at the end.

        Produce the final report using this structure:

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

        IMPORTANT:
        - Use the web search tool before writing the report.
        - Do not fabricate information.
        - Do not fabricate URLs.
        - Base factual claims on information found during research.
        """,

        expected_output=(
            "A professional research report containing an "
            "executive summary, introduction, background, "
            "key findings, detailed analysis, current "
            "developments, challenges and limitations, "
            "conclusion, and source names with URLs."
        ),

        agent=researcher,
    )

    # -----------------------------------------
    # 5. Create Crew
    # -----------------------------------------

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True,
    )

    # -----------------------------------------
    # 6. Run the research crew
    # -----------------------------------------

    result = crew.kickoff()

    return str(result)
