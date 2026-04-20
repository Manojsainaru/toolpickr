"""
Gemini Integration Test for ToolPickr
======================================
Demonstrates the wrap() API — fully automated tool_search orchestration:

  Developer side:
    - Initializes ToolPickr with their chosen embedding provider.
    - Registers all 60+ tools once and calls build().
    - Wraps the Gemini client with pickr.wrap().
    - Calls smart_client.models.generate_content() — ONE call, everything handled.
    - The response behaves like a native Gemini response, plus
      .toolpickr_intercepted and .toolpickr_retrieved_tool_names metadata.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

from toolpickr.pickr import ToolPickr
from tools_definitions import tools

load_dotenv()

# ── Config ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-3.1-flash-lite-preview"

# ── Step 1: Initialize ToolPickr ─────────────────────────────────────────────
print("=" * 60)
print("  ToolPickr — Gemini Integration Test (wrap API)")
print("=" * 60)

pickr = ToolPickr(embedding_provider="google", embedding_model="gemini-embedding-001", embedding_api_key=GEMINI_API_KEY)

print(f"\n[1/4] Registering {len(tools)} tools...")
pickr.register_tools(tools)

print("[2/4] Building FAISS index...")
pickr.build()

# ── Step 2: Wrap the Gemini client ───────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
smart_client = pickr.wrap(gemini_client, format="gemini")
print("[3/4] Wrapped Gemini client.")
print("[4/4] Ready.\n")

# ── Step 3: The agent loop — dramatically simpler ───────────────────────────
def run_agent(user_query: str, existing_system_prompt: str = ""):
    print(f"{'─' * 60}")
    print(f"User: {user_query}")
    print(f"{'─' * 60}")

    # Developer's conversation history — they own this fully
    history = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    # ONE call — ToolPickr handles tool_search interception internally
    response = smart_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=existing_system_prompt,
        )
    )

    # Inspect ToolPickr metadata
    if response.toolpickr_intercepted:
        print(f"  [ToolPickr] Retrieved tools: {response.toolpickr_retrieved_tool_names}")

    # Use the response exactly like a native Gemini response
    part = response.candidates[0].content.parts[0]

    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        print(f"\nGemini called: `{fc.name}`")
        print(f"Arguments:     {dict(fc.args)}\n")
    else:
        print(f"\nGemini: {part.text}\n")


# ── Run test queries ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    MY_SYSTEM_PROMPT = "You are a helpful personal assistant."

    test_queries = [
        "Send an email to john@gmail.com about the project meeting tomorrow.",
        # "What is the current weather in Tokyo?",
        # "Book a flight from New York to London on 2025-05-10.",
        # "What's the stock price for Tesla?",
    ]

    for query in test_queries:
        run_agent(query, existing_system_prompt=MY_SYSTEM_PROMPT)
