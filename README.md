# AI Evals Homework — LangSmith Edition

Companion repo for [Hamel & Shreya's AI Evals course](https://github.com/ai-evals-course/recipe-chatbot). Each homework is reimplemented using **LangSmith** for tracing, evaluation, and monitoring.

All homeworks are built around a **recipe suggestion chatbot** powered by LangChain + Tavily.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- API keys: OpenAI, LangSmith, Tavily

## Setup

```bash
git clone https://github.com/langchain-ai/ai-evals-hw-langsmith.git
cd ai-evals-hw-langsmith

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

## Homeworks

| HW | Topic | Key Concepts |
|----|-------|-------------|
| [HW1](hw1/) | Build & Prompt a Recipe Agent | `create_agent`, Tavily tool, LangSmith tracing |
| [HW2](hw2/) | Error Analysis with Synthetic Data | Annotation queues, Insights reports |
| [HW3](hw3/) | LLM-as-Judge Eval | Datasets, evaluators, `evaluate()`, experiment comparison |
| [HW4](hw4/) | RAG Retrieval Evaluation | Recall@K, MRR, custom evaluators |
| [HW5](hw5/) | Transition Matrix from Errored Traces | Trace upload, SDK trace querying, heatmap visualization |

### HW1: Build & Prompt a Recipe Agent

Build a recipe suggestion bot with LangChain + Tavily. Every invocation is automatically traced to LangSmith. Run 15 diverse test queries and inspect traces in the UI.

### HW2: Error Analysis with Synthetic Data

Import the agent from HW1, run synthetic queries from a CSV, then annotate traces in the LangSmith UI. Use Insights to group and analyze failure modes.

### HW3: LLM-as-Judge Eval

Upload a labeled dataset to LangSmith, create an LLM-as-judge evaluator in the UI, then run two experiments (gpt-4o-mini vs gpt-4o) and compare results.

### HW4: RAG Retrieval Evaluation

Wrap a retrieval pipeline and run it as a LangSmith experiment. Push Recall@K and MRR metrics as custom evaluators. Compare baseline vs larger-K retrieval.

### HW5: Transition Matrix from Errored Traces

Upload synthetic traces with tool call errors to LangSmith. Pull traces via the SDK, build a transition matrix of (last successful step → errored step), and visualize as a heatmap.

## Repo Structure

```
├── hw1/
│   ├── hw1_build_recipe_agent.ipynb
│   └── agent.py                     # Reused by HW2+
├── hw2/
│   ├── hw2_error_analysis.ipynb
│   └── queries.csv                  # Placeholder — fill with your queries
├── hw3/
│   ├── hw3_llm_as_judge.ipynb
│   └── data.csv                     # Labeled examples for dietary compliance
├── hw4/
│   └── hw4_retrieval_eval.ipynb
├── hw5/
│   ├── hw5_transition_matrix.ipynb
│   ├── traces.json                  # Pre-generated synthetic traces
│   └── upload_traces.py             # Uploads traces to LangSmith
├── pyproject.toml
└── .env.example
```
