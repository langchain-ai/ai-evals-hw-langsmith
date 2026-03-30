"""Recipe suggestion agent — reused across all homeworks."""

from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> dict:
    """Search the web for recipes, ingredients, and cooking information."""
    return tavily_client.search(query)


SYSTEM_PROMPT = """\
You are a helpful recipe suggestion assistant. The user will describe what \
ingredients they have, dietary preferences, cuisine interests, or ask general \
cooking questions.

Using the web search tool, find relevant recipes and provide clear, actionable \
suggestions. When recommending a recipe, include:
- Recipe name
- Key ingredients (noting any the user already has)
- Brief preparation overview
- Any relevant dietary notes (allergens, substitutions)

Be friendly, concise, and practical.\
"""


def get_agent(checkpointer=None):
    """Return a configured recipe agent.

    Parameters
    ----------
    checkpointer : optional
        A LangGraph checkpointer for conversation memory.
        Pass ``None`` (default) for stateless, one-shot usage.
    """
    return create_agent(
        model="gpt-4o-mini",
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
