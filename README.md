# 🔎 AI Research Agent

A beginner-friendly single-agent research application built with:

- CrewAI
- Groq
- GPT-OSS 120B
- Serper web search
- Streamlit

## Architecture

User
↓
Streamlit
↓
One CrewAI Research Agent
↓
Serper Web Search
↓
Groq GPT-OSS 120B
↓
Research Report
↓
Streamlit

## Project Structure

```text
ai-research-agent/
│
├── .streamlit/
│   └── secrets.toml.example
│
├── app.py
├── research_agent.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Deployment

This project is designed to be deployed directly to Streamlit Community Cloud through GitHub.

No local `.env` file is required.

### 1. Push this repository to GitHub

Upload the project files to a GitHub repository.

### 2. Create the Streamlit app

In Streamlit Community Cloud, create a new app and select:

- Repository: your GitHub repository
- Branch: main
- Main file: `app.py`

### 3. Add Streamlit Secrets

Open:

`App → Settings → Secrets`

Add:

```toml
GROQ_API_KEY = "your_real_groq_api_key"
SERPER_API_KEY = "your_real_serper_api_key"
```

Save the secrets and deploy/reboot the app.

## Important

Never commit your real API keys to GitHub.

The included `.streamlit/secrets.toml.example` contains placeholders only.

## LLM

The application uses:

```text
groq/openai/gpt-oss-120b
```

## Search

The research agent uses CrewAI's `SerperDevTool`, which requires:

```text
SERPER_API_KEY
```
