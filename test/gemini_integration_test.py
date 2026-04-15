"""
Gemini Integration Test for ToolPickr
======================================
Demonstrates the clean two-turn agentic loop using handle_response():

  Developer side:
    - Initializes ToolPickr with their chosen embedding provider.
    - Registers all 60+ tools once and calls build().
    - Every LLM call passes through pickr.handle_response() — one line.
    - If it was a tool_search, updated_history and retrieved_tools are
      injected into the second call automatically.
    - Conversation history management stays fully in developer hands.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

from toolpickr.pickr import ToolPickr
from toolpickr.embeddings.gemini import GeminiEmbeddings
from tools_definitions import tools

load_dotenv()

# ── Config ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash"
FORMAT         = "gemini"

# ── Step 1: Initialize ToolPickr with the embedding provider ─────────────────
print("=" * 60)
print("  ToolPickr — Gemini Integration Test")
print("=" * 60)

embedding_provider = GeminiEmbeddings(api_key=GEMINI_API_KEY)
pickr = ToolPickr(embedding_provider=embedding_provider)

print(f"\n[1/3] Registering {len(tools)} tools...")
pickr.register_tools(tools)

print("[2/3] Building FAISS index...")
pickr.build()

# ── Step 2: Set up Gemini client ─────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print("[3/3] Ready.\n")

# ── Step 3: The agent loop ───────────────────────────────────────────────────
def run_agent(user_query: str, existing_system_prompt: str = ""):
    print(f"{'─' * 60}")
    print(f"User: {user_query}")
    print(f"{'─' * 60}")

    # Developer's conversation history — they own this fully
    history = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    # ── Turn 1: Only tool_search is provided ─────────────────────────────────
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            # inject_system_prompt auto-appends ToolPickr instructions to the developer's prompt
            system_instruction=pickr.inject_system_prompt(existing_system_prompt),
            # Only one tool given to the LLM at this point
            tools=[pickr.get_search_tool_schema(format=FORMAT)],
        )
    )

    # ── Intercept: one call handles everything ────────────────────────────────
    intercept = pickr.handle_response(response, format=FORMAT)

    if not intercept.is_tool_search:
        # Gemini answered directly without needing a tool — that's fine too
        print(f"\nGemini (direct): {response.candidates[0].content.parts[0].text}\n")
        return

    # ToolPickr handled the tool_search automatically
    # updated_history contains: [model_turn (fc), tool_turn (result)]
    history += intercept.updated_history

    # ── Turn 2: Real tools injected ───────────────────────────────────────────
    response2 = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=pickr.inject_system_prompt(existing_system_prompt),
            tools=intercept.retrieved_tools,   # ← the actual tool schemas
        )
    )

    part2 = response2.candidates[0].content.parts[0]

    if hasattr(part2, "function_call") and part2.function_call:
        fc = part2.function_call
        print(f"\nGemini called: `{fc.name}`")
        print(f"Arguments:     {dict(fc.args)}\n")
    else:
        print(f"\nGemini: {part2.text}\n")


# ── Run test queries ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    MY_SYSTEM_PROMPT = "You are a helpful personal assistant."

    test_queries = [
        "Send an email to john@gmail.com about the project meeting tomorrow.",
        "What is the current weather in Tokyo?",
        "Book a flight from New York to London on 2025-05-10.",
        "What's the stock price for Tesla?",
    ]

    for query in test_queries:
        run_agent(query, existing_system_prompt=MY_SYSTEM_PROMPT)
