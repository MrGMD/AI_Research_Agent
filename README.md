🔎 AI Research Agent

A beginner-friendly single-agent research application built with:

CrewAI

Groq

GPT-OSS 120B

Groq built-in Browser Search

LiteLLM

Streamlit

Architecture

User
  ↓
Streamlit
  ↓
ONE CrewAI Research Agent
  ↓
LiteLLM
  ↓
Groq GPT-OSS 120B
  ↓
Research Report

The agent also has one custom CrewAI tool that calls Groq's built-in
browser_search tool for live web research.

API key

Only one API key is required:

GROQ_API_KEY

There is no Serper API key and no OpenAI API key.

There is also no .env file required.

Model

Groq model:

openai/gpt-oss-120b

CrewAI/LiteLLM model string:

groq/openai/gpt-oss-120b

Streamlit Cloud deployment

Upload/push all project files to GitHub.

Create a Streamlit Community Cloud app.

Set the main file to:

app.py

Use Python 3.12 or 3.13.

Open:

Manage app → Settings → Secrets

Add:

GROQ_API_KEY = "your_real_groq_api_key"

Save and reboot/redeploy the app.

Never commit the real API key to GitHub.

Important compatibility fix

CrewAI 1.15.x can inject a cache_breakpoint property into messages.
Groq rejects that property.

This project therefore disables the CrewAI cache-breakpoint marker and
also removes the property at the LiteLLM boundary before a request is sent
to Groq.

The CrewAI agent also has:

cache=False

Requirements

crewai==1.15.22
groq==1.7.0
litellm==1.102.0
streamlit==1.64.0
