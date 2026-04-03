# AI Evals Homework — LangSmith Edition

Companion repo for [Hamel & Shreya's AI Evals course](https://github.com/ai-evals-course/recipe-chatbot). Each homework is reimplemented using **LangSmith** for tracing, evaluation, and monitoring.

All homeworks are built around a **recipe suggestion chatbot** powered by LangChain.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- API keys: OpenAI (or another model provider), LangSmith, Tavily

## Setup

```bash
git clone https://github.com/langchain-ai/ai-evals-hw-langsmith.git
cd ai-evals-hw-langsmith

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

## Agent Tools - CLI & Skills

Many of the operations used in these notebooks — uploading datasets, running evaluations, querying traces, even creating the original agent — were done with a coding agent using the [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) and [LangSmith Skills](https://github.com/langchain-ai/langsmith-skills). If you prefer a coding agent first workflow, check those out.