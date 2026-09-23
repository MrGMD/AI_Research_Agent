import os

import streamlit as st

from research_agent import run_research


# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)


# -----------------------------------------
# Load API keys from Streamlit Secrets
# -----------------------------------------

try:
    groq_key = st.secrets["GROQ_API_KEY"]
    serper_key = st.secrets["SERPER_API_KEY"]

    # Make them available to the CrewAI module.
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["SERPER_API_KEY"] = serper_key

except KeyError:
    st.error(
        "API keys are not configured. "
        "Please add GROQ_API_KEY and SERPER_API_KEY "
        "in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# -----------------------------------------
# Application UI
# -----------------------------------------

st.title("🔎 AI Research Agent")

st.write(
    "Enter a research topic and the AI agent will "
    "research it using web sources and generate a report."
)


# -----------------------------------------
# Research topic
# -----------------------------------------

topic = st.text_area(
    "Research Topic",
    placeholder=(
        "Example: Impact of Generative AI on software development"
    ),
    height=120,
)


# -----------------------------------------
# Research button
# -----------------------------------------

if st.button(
    "🔍 Start Research",
    type="primary",
    use_container_width=True,
):

    if not topic.strip():
        st.warning("Please enter a research topic.")

    else:
        with st.spinner(
            "Researching the topic... This may take a while."
        ):
            try:
                report = run_research(topic.strip())

                st.success("Research completed!")

                st.markdown(report)

            except Exception as e:
                st.error(
                    "The research agent encountered an error."
                )
                st.exception(e)
