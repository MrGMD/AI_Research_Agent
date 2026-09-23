import os
import streamlit as st

from research_agent import run_research


st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

# Read the API key only from Streamlit Cloud Secrets.
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Go to Streamlit Cloud → Manage app → Settings → Secrets "
        "and add your Groq API key."
    )
    st.stop()

os.environ["GROQ_API_KEY"] = groq_key

st.title("🔎 AI Research Agent")

st.write(
    "Enter a research topic and the AI agent will research it "
    "using Groq GPT-OSS 120B and Groq's built-in browser search."
)

topic = st.text_area(
    "Research Topic",
    placeholder="Example: Impact of Generative AI on software development",
    height=120,
)

if st.button(
    "🔍 Start Research",
    type="primary",
    use_container_width=True,
):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("Researching the topic... This may take a while."):
            try:
                report = run_research(topic.strip())
                st.success("Research completed!")
                st.markdown(report)
            except Exception as e:
                st.error("The research agent encountered an error.")
                st.exception(e)
