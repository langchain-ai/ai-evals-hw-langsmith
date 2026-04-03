"""Recipe suggestion agent — copied from HW1"""

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


recipe_bot = create_agent(
        model="gpt-5.4-nano",
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        name="Recipe_Bot"
    )

recipe_bot_5nano = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        name="Recipe_Bot"
    )