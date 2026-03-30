"""Generate synthetic recipe agent traces with tool call errors.

Produces traces.json — a flat list of runs consumable by upload_traces.py.

~60 traces total: 35 successful, 25 with errored tool calls.
Each trace: root chain → LLM call → tool call → (optional retry) → final LLM response.
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone

MODEL_NAME = "gpt-4o-mini"
SEED = 42
OUTPUT_PATH = "hw5/traces.json"

SYSTEM_MESSAGE = (
    "You are a helpful recipe suggestion assistant. The user will describe what "
    "ingredients they have, dietary preferences, cuisine interests, or ask general "
    "cooking questions. Using the web search tool, find relevant recipes and provide "
    "clear, actionable suggestions."
)

RECIPE_QUERIES = [
    "I have chicken thighs, rice, and bell peppers. What can I make?",
    "Suggest a vegan dinner I can make in under 30 minutes.",
    "I'm gluten-free and craving Italian food. Suggestions?",
    "Give me a keto-friendly lunch recipe with ground beef.",
    "I only have eggs, butter, and flour. What can I bake?",
    "Suggest a Thai curry recipe for beginners.",
    "What's a healthy meal-prep recipe for the whole week?",
    "I have a bunch of zucchini from my garden. What should I do with it?",
    "Recommend a kid-friendly dinner that's also nutritious.",
    "I want to make a fancy dessert for a dinner party. Any ideas?",
    "What's a traditional Mexican breakfast recipe?",
    "I'm dairy-free and want mac and cheese. How?",
    "Suggest a high-protein vegetarian recipe.",
    "I have canned chickpeas, spinach, and garlic. What can I make?",
    "What's an easy recipe for homemade pasta sauce?",
    "How do I make chicken tikka masala from scratch?",
    "What are good vegan protein sources for cooking?",
    "Best way to bake sourdough bread at home?",
    "Quick weeknight pasta recipes?",
    "Gluten-free dessert ideas?",
    "I have leftover salmon and cream cheese. Ideas?",
    "What's a good recipe for homemade ramen?",
    "Suggest Korean side dishes (banchan) I can make.",
    "How do I properly sear a steak?",
    "What is the best way to caramelize onions?",
    "How do I make a roux for gumbo?",
    "What temperature should I roast vegetables at?",
    "Easy one-pot dinner recipes?",
    "What are good freezer-friendly meals?",
    "Suggest warm comfort food for winter.",
    "What are festive holiday cookie recipes?",
    "Easy appetizers for a cocktail party?",
    "Suggest a brunch menu for 8 people.",
    "What's a good recipe for Greek moussaka?",
    "How do I make Moroccan tagine with lamb?",
    "I have tofu, soy sauce, and broccoli. Quick stir-fry recipe?",
    "Suggest a low-calorie dinner with salmon.",
    "What's a good substitute for eggs in baking?",
    "How do I make fluffy pancakes from scratch?",
    "Recommend a recipe using canned tuna.",
    "I'm allergic to nuts. Give me a Thai curry recipe.",
    "What are good packed lunch ideas for work?",
    "Suggest romantic dinner ideas for two.",
    "What should I make for a summer BBQ?",
    "How do I make homemade pizza dough?",
    "What are easy slow cooker dump meals?",
    "Suggest a birthday dinner menu.",
    "Quick breakfast ideas for busy mornings?",
    "I have sweet potatoes and black beans. What can I cook?",
    "How do I make a proper French omelette?",
    "What's a good recipe for banana bread?",
    "Suggest Mediterranean diet lunch ideas.",
    "How do I cook perfect jasmine rice?",
    "What are good picnic food ideas?",
    "I need a nut-free school snack for my kid.",
    "How do I make authentic guacamole?",
    "Suggest a vegetarian Thanksgiving main course.",
    "What's a good marinade for grilled chicken?",
    "How do I make cold brew coffee at home?",
    "Suggest an easy fish taco recipe.",
]

SEARCH_RESULTS = [
    "Found 8 recipes: Mediterranean Bowl (25 min), Sheet Pan Fajitas (30 min), One-Pot Pasta (20 min). Source: allrecipes.com",
    "Top results: Classic Stir Fry (15 min), Baked Salmon (25 min), Stuffed Peppers (40 min). Source: seriouseats.com",
    "Trending: Creamy Garlic Chicken (30 min), Lemon Herb Roasted Veggies (35 min), Quick Tacos (15 min). Source: bonappetit.com",
    "Community favorites: Mushroom Risotto (40 min), Thai Basil Noodles (20 min), Caprese Salad (10 min). Source: epicurious.com",
    "Popular recipes: Black Bean Soup (35 min), Veggie Curry (30 min), Grilled Cheese & Tomato Soup (20 min). Source: foodnetwork.com",
    "Recommended: Shakshuka (25 min), Avocado Toast Variations (10 min), Overnight Oats (5 min). Source: minimalistbaker.com",
    "Best rated: Beef Stew (2 hrs), Chicken Pot Pie (50 min), Mac and Cheese (25 min). Source: allrecipes.com",
    "Chef picks: Homemade Pizza (45 min), Caesar Salad (15 min), Garlic Bread (10 min). Source: foodnetwork.com",
]

LLM_RESPONSES = [
    "Great question! Based on my search, here are some excellent options:\n\n{result}\n\nWould you like the full recipe for any of these?",
    "I found some wonderful recipes for you!\n\n{result}\n\nLet me know if you'd like more details.",
    "Here are my top recommendations:\n\n{result}\n\nI can provide step-by-step instructions for whichever catches your eye!",
    "Perfect timing — here's what I found:\n\n{result}\n\nWant me to adjust any of these for dietary preferences?",
    "I've got you covered! Check these out:\n\n{result}\n\nI can also suggest ingredient substitutions if needed.",
]

FALLBACK_RESPONSES = [
    "I wasn't able to search for recipes right now, but based on my knowledge, here are some suggestions:\n\n{result}",
    "The recipe search is temporarily unavailable, but I can still help! From what I know:\n\n{result}",
    "Having trouble reaching the recipe database, but here's what I can suggest:\n\n{result}",
]

TOOL_ERRORS = [
    "ConnectionTimeoutError: Request to recipe search API timed out after 30s",
    "HTTPError: 503 Service Unavailable - recipe search backend is temporarily down",
    "ConnectionError: Failed to establish connection to search.recipes.api",
    "TimeoutError: web_search exceeded maximum execution time of 15000ms",
    "HTTPError: 502 Bad Gateway - upstream recipe service returned an invalid response",
    "ConnectionResetError: Connection was reset by the remote recipe API server",
    "HTTPError: 429 Too Many Requests - rate limit exceeded for recipe search API",
]


def _uuid():
    return str(uuid.uuid4())


def _llm_inputs(query):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": query},
        ]
    }


def _tool_call_obj(query):
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": query})},
    }


def _llm_outputs(text, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [
            {"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}
        ]
    }


def build_success_trace(query, base_time):
    """Build a trace where the tool call succeeds."""
    trace_id = _uuid()
    root_id = _uuid()
    llm1_id = _uuid()
    tool_id = _uuid()
    llm2_id = _uuid()

    root_start = base_time
    llm1_start = root_start + timedelta(milliseconds=random.randint(5, 30))
    llm1_end = llm1_start + timedelta(milliseconds=random.randint(300, 800))
    tool_start = llm1_end + timedelta(milliseconds=random.randint(5, 20))
    tool_end = tool_start + timedelta(milliseconds=random.randint(200, 1500))
    llm2_start = tool_end + timedelta(milliseconds=random.randint(5, 20))
    llm2_end = llm2_start + timedelta(milliseconds=random.randint(400, 1200))
    root_end = llm2_end + timedelta(milliseconds=random.randint(5, 30))

    search_result = random.choice(SEARCH_RESULTS)
    final_answer = random.choice(LLM_RESPONSES).format(result=search_result)
    tc_obj = _tool_call_obj(query)

    metadata = {"agent_version": "1.0.0"}
    tags = ["recipe-agent", "production"]

    runs = []

    # Root chain
    runs.append({
        "id": root_id,
        "trace_id": trace_id,
        "parent_run_id": None,
        "name": "Recipe Agent",
        "run_type": "chain",
        "inputs": {"question": query},
        "outputs": {"output": final_answer},
        "error": None,
        "extra": {"metadata": metadata},
        "tags": tags,
        "start_time": root_start.isoformat(),
        "end_time": root_end.isoformat(),
    })

    # LLM 1 — tool-calling
    runs.append({
        "id": llm1_id,
        "trace_id": trace_id,
        "parent_run_id": root_id,
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": _llm_inputs(query),
        "outputs": _llm_outputs("", [tc_obj]),
        "error": None,
        "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
        "tags": tags,
        "start_time": llm1_start.isoformat(),
        "end_time": llm1_end.isoformat(),
    })

    # Tool — success
    runs.append({
        "id": tool_id,
        "trace_id": trace_id,
        "parent_run_id": root_id,
        "name": "web_search",
        "run_type": "tool",
        "inputs": {"query": query},
        "outputs": {"result": search_result},
        "error": None,
        "extra": {"metadata": metadata},
        "tags": tags + ["tool"],
        "start_time": tool_start.isoformat(),
        "end_time": tool_end.isoformat(),
    })

    # LLM 2 — final answer
    runs.append({
        "id": llm2_id,
        "trace_id": trace_id,
        "parent_run_id": root_id,
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": _llm_inputs(query),
        "outputs": _llm_outputs(final_answer),
        "error": None,
        "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
        "tags": tags,
        "start_time": llm2_start.isoformat(),
        "end_time": llm2_end.isoformat(),
    })

    return runs


def build_error_trace(query, base_time, with_retry=True):
    """Build a trace where the tool call fails, optionally with a retry."""
    trace_id = _uuid()
    root_id = _uuid()
    llm1_id = _uuid()
    tool_id = _uuid()

    root_start = base_time
    llm1_start = root_start + timedelta(milliseconds=random.randint(5, 30))
    llm1_end = llm1_start + timedelta(milliseconds=random.randint(300, 800))
    tool_start = llm1_end + timedelta(milliseconds=random.randint(5, 20))
    tool_end = tool_start + timedelta(milliseconds=random.randint(2000, 5000))

    error_msg = random.choice(TOOL_ERRORS)
    tc_obj = _tool_call_obj(query)
    metadata = {"agent_version": "1.0.0"}

    runs = []

    if with_retry:
        # Retry LLM + retry tool (success) + final LLM
        retry_llm_id = _uuid()
        retry_tool_id = _uuid()
        final_llm_id = _uuid()

        retry_llm_start = tool_end + timedelta(milliseconds=random.randint(50, 200))
        retry_llm_end = retry_llm_start + timedelta(milliseconds=random.randint(300, 600))
        retry_tool_start = retry_llm_end + timedelta(milliseconds=random.randint(5, 20))
        retry_tool_end = retry_tool_start + timedelta(milliseconds=random.randint(200, 1500))
        final_llm_start = retry_tool_end + timedelta(milliseconds=random.randint(5, 20))
        final_llm_end = final_llm_start + timedelta(milliseconds=random.randint(400, 1200))
        root_end = final_llm_end + timedelta(milliseconds=random.randint(5, 30))

        search_result = random.choice(SEARCH_RESULTS)
        final_answer = random.choice(LLM_RESPONSES).format(result=search_result)
        retry_tc_obj = _tool_call_obj(query)
        tags = ["recipe-agent", "production", "tool-error-recovered"]

        # Root
        runs.append({
            "id": root_id, "trace_id": trace_id, "parent_run_id": None,
            "name": "Recipe Agent", "run_type": "chain",
            "inputs": {"question": query}, "outputs": {"output": final_answer},
            "error": None, "extra": {"metadata": metadata}, "tags": tags,
            "start_time": root_start.isoformat(), "end_time": root_end.isoformat(),
        })

        # LLM 1 — tool-calling
        runs.append({
            "id": llm1_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "ChatOpenAI", "run_type": "llm",
            "inputs": _llm_inputs(query), "outputs": _llm_outputs("", [tc_obj]),
            "error": None,
            "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
            "tags": ["recipe-agent"], "start_time": llm1_start.isoformat(), "end_time": llm1_end.isoformat(),
        })

        # Tool — error
        runs.append({
            "id": tool_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "web_search", "run_type": "tool",
            "inputs": {"query": query}, "outputs": None,
            "error": error_msg, "extra": {"metadata": metadata},
            "tags": ["recipe-agent", "tool", "error"],
            "start_time": tool_start.isoformat(), "end_time": tool_end.isoformat(),
        })

        # Retry LLM — decides to retry
        runs.append({
            "id": retry_llm_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "ChatOpenAI", "run_type": "llm",
            "inputs": _llm_inputs(query), "outputs": _llm_outputs("", [retry_tc_obj]),
            "error": None,
            "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
            "tags": ["recipe-agent", "retry"],
            "start_time": retry_llm_start.isoformat(), "end_time": retry_llm_end.isoformat(),
        })

        # Retry tool — success
        runs.append({
            "id": retry_tool_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "web_search", "run_type": "tool",
            "inputs": {"query": query}, "outputs": {"result": search_result},
            "error": None, "extra": {"metadata": metadata},
            "tags": ["recipe-agent", "tool", "retry"],
            "start_time": retry_tool_start.isoformat(), "end_time": retry_tool_end.isoformat(),
        })

        # Final LLM — answer
        runs.append({
            "id": final_llm_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "ChatOpenAI", "run_type": "llm",
            "inputs": _llm_inputs(query), "outputs": _llm_outputs(final_answer),
            "error": None,
            "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
            "tags": ["recipe-agent"],
            "start_time": final_llm_start.isoformat(), "end_time": final_llm_end.isoformat(),
        })

    else:
        # No retry — LLM falls back to knowledge
        fallback_llm_id = _uuid()
        fallback_llm_start = tool_end + timedelta(milliseconds=random.randint(50, 200))
        fallback_llm_end = fallback_llm_start + timedelta(milliseconds=random.randint(400, 1200))
        root_end = fallback_llm_end + timedelta(milliseconds=random.randint(5, 30))

        fallback_answer = random.choice(FALLBACK_RESPONSES).format(result=random.choice(SEARCH_RESULTS))
        tags = ["recipe-agent", "production", "tool-error-fallback"]

        # Root
        runs.append({
            "id": root_id, "trace_id": trace_id, "parent_run_id": None,
            "name": "Recipe Agent", "run_type": "chain",
            "inputs": {"question": query}, "outputs": {"output": fallback_answer},
            "error": None, "extra": {"metadata": metadata}, "tags": tags,
            "start_time": root_start.isoformat(), "end_time": root_end.isoformat(),
        })

        # LLM 1 — tool-calling
        runs.append({
            "id": llm1_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "ChatOpenAI", "run_type": "llm",
            "inputs": _llm_inputs(query), "outputs": _llm_outputs("", [tc_obj]),
            "error": None,
            "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
            "tags": ["recipe-agent"], "start_time": llm1_start.isoformat(), "end_time": llm1_end.isoformat(),
        })

        # Tool — error
        runs.append({
            "id": tool_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "web_search", "run_type": "tool",
            "inputs": {"query": query}, "outputs": None,
            "error": error_msg, "extra": {"metadata": metadata},
            "tags": ["recipe-agent", "tool", "error"],
            "start_time": tool_start.isoformat(), "end_time": tool_end.isoformat(),
        })

        # Fallback LLM — answer from knowledge
        runs.append({
            "id": fallback_llm_id, "trace_id": trace_id, "parent_run_id": root_id,
            "name": "ChatOpenAI", "run_type": "llm",
            "inputs": _llm_inputs(query), "outputs": _llm_outputs(fallback_answer),
            "error": None,
            "extra": {"metadata": {**metadata, "ls_model_name": MODEL_NAME, "ls_model_type": "chat"}},
            "tags": ["recipe-agent", "fallback"],
            "start_time": fallback_llm_start.isoformat(), "end_time": fallback_llm_end.isoformat(),
        })

    return runs


def generate_traces(seed=SEED):
    random.seed(seed)
    all_runs = []
    base_time = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)

    queries = list(RECIPE_QUERIES)
    random.shuffle(queries)

    # 35 successful traces
    for i in range(35):
        query = queries[i % len(queries)]
        trace_time = base_time + timedelta(
            minutes=random.randint(0, 180),
            seconds=random.randint(0, 59),
        )
        all_runs.extend(build_success_trace(query, trace_time))

    # 25 error traces (~60% retry, ~40% fallback)
    for i in range(25):
        query = queries[(35 + i) % len(queries)]
        trace_time = base_time + timedelta(
            minutes=random.randint(0, 180),
            seconds=random.randint(0, 59),
        )
        with_retry = random.random() < 0.6
        all_runs.extend(build_error_trace(query, trace_time, with_retry=with_retry))

    return all_runs


def main():
    runs = generate_traces()

    trace_ids = {r["trace_id"] for r in runs}
    tool_runs = [r for r in runs if r["run_type"] == "tool"]
    errored_tools = [r for r in tool_runs if r.get("error")]

    print(f"Generated {len(runs)} runs across {len(trace_ids)} traces")
    print(f"  Tool calls: {len(tool_runs)} total, {len(errored_tools)} errored ({len(errored_tools)/max(len(tool_runs),1)*100:.0f}%)")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(runs, f, indent=2, default=str)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
