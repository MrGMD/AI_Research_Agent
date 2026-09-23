🔎 AI Research Agent

Beginner-friendly single-agent research app using CrewAI, Groq GPT-OSS 120B, Groq built-in Browser Search, LiteLLM, and Streamlit.

API key

Only one secret is required:

GROQ_API_KEY

No Serper API key and no .env file are required.

Architecture

User
→ Streamlit
→ One CrewAI Research Agent
→ LiteLLM
→ Groq GPT-OSS 120B

The agent also has a custom CrewAI web-research tool that calls Groq's built-in Browser Search using the same GROQ_API_KEY.

Deployment

Push these files to GitHub.

Create a Streamlit Community Cloud app using app.py.

Use Python 3.12 or 3.13.

In Streamlit Cloud:
Manage app → Settings → Secrets

Add:

GROQ_API_KEY = "your_real_groq_api_key"

Save and redeploy/reboot the app.

Never put the real API key in GitHub or Python source files.

Model

openai/gpt-oss-120b

For CrewAI/LiteLLM, the model is specified as:

groq/openai/gpt-oss-120b

Dependencies

crewai==1.15.22
groq==1.7.0
litellm==1.102.0
streamlit==1.64.0
