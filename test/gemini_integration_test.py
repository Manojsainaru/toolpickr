"""
Gemini Integration Test for ToolPickr
======================================
Demonstrates the new universal tool architecture:

  Developer side:
    - Initializes ToolPickr with their chosen embedding provider.
    - Registers all 60+ tools once and calls build().
    - Gets the 'toolpickr' tool schema + system prompt.
    - Makes standard Gemini API calls — ToolPickr is just another tool.
    - Handles toolpickr function calls in their own loop using pickr.handle_tool_call().

  The user has FULL control over the LLM loop. No wrappers, no proxies.
  ToolPickr is just a tool that the LLM calls like any other.
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
print("  ToolPickr — Gemini Integration Test (universal tool)")
print("=" * 60)

pickr = ToolPickr(
    embedding_provider="google",
    embedding_model="gemini-embedding-001",
    embedding_api_key=GEMINI_API_KEY,
    auto_execute=False,  # Router mode — we handle execution ourselves
)

print(f"\n[1/4] Registering {len(tools)} tools...")
pickr.register_tools(tools)

print("[2/4] Building FAISS index...")
pickr.build()

# ── Step 2: Get toolpickr tool + system prompt ──────────────────────────────
toolpickr_tool = pickr.get_tool(format="gemini")
print("[3/4] Got toolpickr tool schema.")
print("[4/4] Ready.\n")


# ── Step 3: The agent loop — user has full control ──────────────────────────
def run_agent(user_query: str, existing_system_prompt: str = ""):
    print(f"{'─' * 60}")
    print(f"User: {user_query}")
    print(f"{'─' * 60}")

    # Merge system prompts
    system_prompt = pickr.get_system_prompt(existing_system_prompt)

    # Developer's conversation history — they own this fully
    history = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    # Config with toolpickr as a tool (user can add their own tools too)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[toolpickr_tool],  # + any other tools the user wants
    )

    # Standard Gemini client — no wrapping
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Agent loop: handle toolpickr calls ──
    max_turns = 5  # Safety limit
    for turn in range(max_turns):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=config,
        )

        # Get the response part
        candidate = response.candidates[0]
        
        # Append the model response to history
        history.append(candidate.content)

        # Check for function calls
        function_call = None
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_call = part.function_call
                break

        if function_call is None:
            # No function call — model gave a text response, we're done
            print(f"\nGemini: {candidate.content.parts[0].text}\n")
            return

        print(f"\n  [Turn {turn + 1}] LLM called: {function_call.name}({dict(function_call.args)})")

        if function_call.name == "toolpickr":
            # Handle the toolpickr call
            result = pickr.handle_tool_call(dict(function_call.args))

            if result.action == "search":
                tool_names = [t["name"] for t in result.data.get("available_tools", [])]
                print(f"  [ToolPickr] Found tools: {tool_names}")
                
                # Feed the search results back to the LLM
                response_data = result.data

            elif result.action == "execute":
                if result.executed:
                    # auto_execute was on and handler existed
                    print(f"  [ToolPickr] Executed '{result.tool_name}': {result.data}")
                    response_data = result.data
                else:
                    # Router mode — we need to execute it ourselves
                    print(f"  [ToolPickr] Route to: {result.tool_name}({result.data.get('tool_arguments', {})})")
                    # Here the developer would call their actual tool implementation:
                    # actual_result = my_tools[result.tool_name](**result.data["tool_arguments"])
                    # For this demo, we simulate a response:
                    response_data = {"result": f"[SIMULATED] Tool '{result.tool_name}' executed successfully with args: {result.data.get('tool_arguments', {})}"}
            else:
                response_data = {"error": result.error}

            # Append tool response to history
            history.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="toolpickr",
                                response=response_data,
                            )
                        )
                    ]
                )
            )

        else:
            # Non-toolpickr function call — handle normally
            print(f"\n  Gemini called external tool: `{function_call.name}`")
            print(f"  Arguments: {dict(function_call.args)}\n")
            return

    print("\n[Agent] Max turns reached.\n")


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
