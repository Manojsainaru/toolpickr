<p align="center">
  <h1 align="center">🔧 ToolPickr</h1>
  <p align="center"><strong>Smart and scalable tool picking for LLMs</strong></p>
  <p align="center">
    Give your LLM access to hundreds of tools — ToolPickr finds the right ones instantly.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#installation">Installation</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#examples">Examples</a>
</p>

---

## The Problem

LLMs have a limited context window. When you have 100+ tools, you can't pass all of them in every API call — it's expensive, slow, and degrades tool selection accuracy.

## The Solution

**ToolPickr** is a retrieval-augmented tool picker. It uses an **ensemble of semantic search + BM25** (with an optional **cross-encoder reranker**) to find the most relevant tools for each query. Your LLM only sees the tools it actually needs.

ToolPickr exposes itself as a **single meta-tool** called `toolpickr` that your LLM calls in a two-step protocol:

1. **Search** — LLM describes what it needs → ToolPickr returns matching tools
2. **Execute** — LLM calls a discovered tool → ToolPickr routes (or auto-executes) it

You keep **full control** of your LLM loop. No wrappers, no proxies, no magic.

---

## Quick Start

```python
from toolpickr import ToolPickr, ToolDefinition

# 1. Initialize — zero-config works out of the box (local models, no API key)
pickr = ToolPickr()

# 2. Register your tools
pickr.register_tools([
    ToolDefinition(
        name="send_email",
        description="Send an email to a recipient.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
            },
            "required": ["to", "subject", "body"]
        }
    ),
    ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location name"},
            },
            "required": ["location"]
        }
    ),
    # ... register 100+ tools
])

# 3. Build the search index
pickr.build()

# 4. Get the toolpickr tool schema for your LLM
tool_schema = pickr.get_tool(format="gemini")  # or format="base" for raw dict
system_prompt = pickr.get_system_prompt("You are a helpful assistant.")

# 5. In your LLM tool-call loop, handle toolpickr calls:
result = pickr.handle_tool_call({"action": "search", "queries": ["send an email"]})
# result.data["available_tools"] → list of matching tool schemas
```

---

## Installation

```bash
# Core (includes sentence-transformers for local embeddings + BM25)
pip install toolpickr

# With Google Gemini support
pip install toolpickr[google]

# With OpenAI support
pip install toolpickr[openai]

# Everything
pip install toolpickr[all]
```

### Requirements

- Python ≥ 3.9
- Core dependencies: `faiss-cpu`, `numpy`, `pydantic`, `rank-bm25`, `sentence-transformers`

---

## How It Works

### Architecture

```
Your LLM ←→ toolpickr (meta-tool)
                │
                ├── action="search"  → Ensemble Retriever → [Reranker] → tool schemas
                │                       ├── Semantic (FAISS cosine similarity)
                │                       └── BM25 (lexical matching)
                │
                └── action="execute" → Route to handler (or return call info)
```

### Retrieval Pipeline

| Stage | What It Does | Config |
|-------|-------------|--------|
| **Semantic** | Embeds query → FAISS cosine similarity over tool embeddings | `embedding_model`, `semantic_weight` |
| **BM25** | Lexical keyword matching over tool descriptions | `bm25_weight` |
| **Ensemble** | Min-max normalizes + weighted fusion of both scores | `semantic_weight`, `bm25_weight` |
| **Reranker** *(optional)* | Cross-encoder re-scores top candidates | `reranker_model`, `top_p` |
| **Final** | Returns the best `top_k` tools | `top_k` |

### Two-Step Protocol

When your LLM receives a user request, it calls the `toolpickr` tool:

**Step 1 — Search:**
```json
{
  "action": "search",
  "queries": ["send an email", "get the weather"]
}
```
ToolPickr returns matching tool schemas with their parameter definitions.

**Step 2 — Execute:**
```json
{
  "action": "execute",
  "tool_name": "send_email",
  "tool_arguments": {"to": "john@example.com", "subject": "Hi", "body": "Hello!"}
}
```
ToolPickr either auto-executes (if a handler is registered) or returns the call info for you to handle.

---

## Configuration

### Constructor Parameters

```python
pickr = ToolPickr(
    # ── Embedding Model ──
    embedding_model="sentence_transformers/all-MiniLM-L6-v2",  # "provider/model" format
    embedding_api_key=None,       # Optional — auto-discovered from env vars

    # ── Reranker (optional) ──
    reranker_model=None,          # e.g., "cross_encoder/ms-marco-MiniLM-L-6-v2"
    reranker_api_key=None,        # Optional — auto-discovered from env vars

    # ── Retrieval Tuning ──
    top_k=5,                      # Final number of tools returned per query
    top_p=15,                     # Candidates before reranking (only with reranker)
    semantic_weight=0.5,          # Weight for embedding similarity [0.0 - 1.0]
    bm25_weight=0.5,              # Weight for BM25 lexical score [0.0 - 1.0]

    # ── Execution ──
    auto_execute=False,           # True = execute tools with handlers automatically

    # ── Debug ──
    debug=False,                  # True = print detailed pipeline logs
)
```

### Embedding Models

Models use the `"provider/model"` format. API keys are auto-discovered from environment variables.

| Model String | Provider | Env Var | Notes |
|-------------|----------|---------|-------|
| `sentence_transformers/all-MiniLM-L6-v2` | Local | *None needed* | **Default**. Fast, 384 dims |
| `sentence_transformers/all-mpnet-base-v2` | Local | *None needed* | Higher quality, 768 dims |
| `google/gemini-embedding-001` | Google | `GEMINI_API_KEY` | 3072 dims, best quality |
| `google/text-embedding-004` | Google | `GEMINI_API_KEY` | 768 dims |
| `openai/text-embedding-3-small` | OpenAI | `OPENAI_API_KEY` | 1536 dims, cheapest |
| `openai/text-embedding-3-large` | OpenAI | `OPENAI_API_KEY` | 3072 dims, best quality |

### Reranker Models

| Model String | Provider | Notes |
|-------------|----------|-------|
| `cross_encoder/ms-marco-MiniLM-L-6-v2` | Local | Fast, lightweight |
| `cross_encoder/ms-marco-MiniLM-L-12-v2` | Local | More accurate |

---

## Examples

### Gemini Integration (Full Example)

```python
import os
from google import genai
from google.genai import types
from toolpickr import ToolPickr, ToolDefinition

# Initialize with Google embeddings (API key from GEMINI_API_KEY env var)
pickr = ToolPickr(
    embedding_model="google/gemini-embedding-001",
    auto_execute=False,
)

# Register tools
pickr.register_tools(my_tools)  # your list of ToolDefinition objects
pickr.build()

# Get the toolpickr schema formatted for Gemini
toolpickr_tool = pickr.get_tool(format="gemini")
system_prompt = pickr.get_system_prompt("You are a helpful assistant.")

# Standard Gemini API — you own the loop
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    tools=[toolpickr_tool],
)

history = [types.Content(role="user", parts=[types.Part(text="Send an email to john@example.com")])]

for turn in range(5):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=config,
    )

    candidate = response.candidates[0]
    history.append(candidate.content)

    # Check for function calls
    function_call = None
    for part in candidate.content.parts:
        if hasattr(part, "function_call") and part.function_call:
            function_call = part.function_call
            break

    if not function_call:
        print(f"Response: {candidate.content.parts[0].text}")
        break

    if function_call.name == "toolpickr":
        result = pickr.handle_tool_call(dict(function_call.args))

        if result.action == "search":
            response_data = result.data
        elif result.action == "execute":
            if result.executed:
                response_data = result.data
            else:
                # Route to your tool implementation
                response_data = my_tool_handlers[result.tool_name](**result.data["tool_arguments"])

        history.append(types.Content(
            role="tool",
            parts=[types.Part(function_response=types.FunctionResponse(
                name="toolpickr", response=response_data,
            ))]
        ))
```

### Zero-Config (Local, No API Key)

```python
from toolpickr import ToolPickr

# Uses sentence-transformers locally — no API key needed
pickr = ToolPickr()
pickr.register_tools(my_tools)
pickr.build()

# Direct search (no LLM needed)
results = pickr.retrieve("send an email")
for r in results:
    print(f"  {r.tool_name}: {r.score:.4f}")
```

### With Reranker

```python
pickr = ToolPickr(
    embedding_model="google/gemini-embedding-001",
    reranker_model="cross_encoder/ms-marco-MiniLM-L-6-v2",
    top_k=3,     # Final 3 tools
    top_p=15,    # Reranker picks from top 15 ensemble candidates
)
```

### Auto-Execute Mode

```python
def send_email(to: str, subject: str, body: str):
    print(f"Sending email to {to}: {subject}")
    return {"status": "sent"}

pickr = ToolPickr(auto_execute=True)

# Register tool with its handler
pickr.register_tool(
    ToolDefinition(name="send_email", description="Send an email", parameters=...),
    handler=send_email,
)

pickr.build()

# When the LLM calls execute, ToolPickr runs the handler automatically
result = pickr.handle_tool_call({
    "action": "execute",
    "tool_name": "send_email",
    "tool_arguments": {"to": "john@example.com", "subject": "Hi", "body": "Hello!"}
})

print(result.executed)  # True
print(result.data)      # {"result": {"status": "sent"}}
```

### Custom Retrieval Weights

```python
# Favor semantic similarity (good for descriptive queries)
pickr = ToolPickr(semantic_weight=0.7, bm25_weight=0.3)

# Favor keyword matching (good for exact tool name queries)
pickr = ToolPickr(semantic_weight=0.3, bm25_weight=0.7)
```

---

## API Reference

### `ToolPickr`

| Method | Description |
|--------|-------------|
| `register_tool(tool, handler=None)` | Register a single tool with optional callable handler |
| `register_tools(tools)` | Bulk-register a list of tools |
| `register_handler(tool_name, handler)` | Add a handler to an already-registered tool |
| `build()` | Build the FAISS + BM25 search indexes |
| `get_tool(format="base")` | Get the `toolpickr` meta-tool schema (`"base"` or `"gemini"`) |
| `get_system_prompt(existing="")` | Get the system prompt with ToolPickr instructions |
| `handle_tool_call(args)` | Process a toolpickr function call from the LLM → `ToolCallResult` |
| `retrieve(query, top_k=None)` | Direct retrieval — returns `SearchResult` objects |

### `ToolDefinition`

Pydantic model for defining tools:

```python
ToolDefinition(
    name="tool_name",              # Required: unique tool name
    description="What it does",     # Required: natural language description
    parameters=Parameters(          # Optional: JSON Schema for arguments
        type="object",
        properties={
            "param_name": Property(type="string", description="..."),
        },
        required=["param_name"]
    ),
    category="optional_category",   # Optional: used in embeddings
)
```

### `ToolCallResult`

Returned by `handle_tool_call()`:

| Field | Type | Description |
|-------|------|-------------|
| `action` | `str` | `"search"` or `"execute"` |
| `success` | `bool` | Whether the operation succeeded |
| `data` | `dict` | Response payload (tool schemas or execution result) |
| `tool_name` | `str \| None` | Tool name (for execute actions) |
| `error` | `str \| None` | Error message if `success=False` |
| `executed` | `bool` | `True` if tool was auto-executed, `False` if routed |

---

## Project Structure

```
toolpickr/
├── __init__.py             # Public API exports
├── _version.py             # Package version
├── pickr.py                # Main facade — ToolPickr class
├── core/
│   ├── tool.py             # ToolDefinition, Parameters, Property (Pydantic)
│   ├── registry.py         # ToolRegistry — stores tools + optional handlers
│   └── results.py          # ToolCallResult dataclass
├── embeddings/
│   ├── base.py             # EmbeddingProvider ABC
│   ├── factory.py          # create_embedding_provider("provider/model")
│   ├── google.py           # Google Gemini embeddings
│   ├── openai.py           # OpenAI embeddings
│   ├── sentence_transformers.py  # Local sentence-transformers
│   └── renderer.py         # Renders ToolDefinition → text for embedding
├── vectorstores/
│   ├── base.py             # VectorStore ABC + SearchResult
│   └── faiss.py            # FAISS vector store with disk persistence
├── retrieval/
│   ├── flat.py             # Semantic retriever (embed query → FAISS search)
│   ├── bm25.py             # BM25 lexical retriever
│   ├── ensemble.py         # Weighted fusion of semantic + BM25
│   └── reranker.py         # Cross-encoder reranker
└── integrations/
    └── gemini.py           # Tool schema converter for Gemini
```

---

## Environment Variables

ToolPickr auto-discovers API keys from environment variables:

| Variable | Used By |
|----------|---------|
| `GEMINI_API_KEY` | Google embeddings (`google/...`) |
| `GOOGLE_API_KEY` | Google embeddings (fallback) |
| `OPENAI_API_KEY` | OpenAI embeddings (`openai/...`) |

You can always override with explicit `embedding_api_key` or `reranker_api_key` parameters.

---

## LLM Integration Support

| LLM Provider | Tool Schema Format | Status |
|-------------|-------------------|--------|
| Google Gemini | `get_tool(format="gemini")` | ✅ Supported |
| OpenAI / GPT | `get_tool(format="base")` | 🔜 Coming soon |
| Any LLM | `get_tool(format="base")` | ✅ Use raw dict schema |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
