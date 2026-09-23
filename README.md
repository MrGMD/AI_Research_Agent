🔎 AI Research Agent

Beginner-friendly single-agent research app using CrewAI, Groq GPT-OSS 120B, Groq built-in Browser Search, and Streamlit.

One API key only

GROQ_API_KEY

No Serper API key and no .env file are required.

Architecture

User → Streamlit → One CrewAI Research Agent → Groq Web Research → Groq Browser Search → Research Report

Deployment

Push these files to GitHub.

Create the Streamlit Community Cloud app with app.py.

Select Python 3.13. CrewAI 1.15.22 requires Python >=3.10 and <3.14.

In Manage app → Settings → Secrets, add:

GROQ_API_KEY = "your_real_groq_api_key"

Save and deploy/reboot.

Never put the real API key in GitHub or Python source files.

Model

openai/gpt-oss-120b

Dependencies

crewai==1.15.22
groq==1.7.0
streamlit==1.64.0
