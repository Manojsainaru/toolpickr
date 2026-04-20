# ToolPickr — Full System Architecture Document

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER APPLICATION                                   │
│          (Direct Python API / LangChain / LangGraph / OpenAI FC)                │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ToolPickr  (Facade)                                    │
│                                                                                 │
│  ┌──────────┐  ┌───────────────┐  ┌───────────┐  ┌────────────┐  ┌──────────┐ │
│  │  Config   │  │  Tool         │  │ Categori- │  │ Retrieval  │  │ LLM      │ │
│  │  Manager  │  │  Registry     │  │ zation    │  │ Engine     │  │ Selector │ │
│  └──────────┘  └───────────────┘  │ Engine    │  └─────┬──────┘  └──────────┘ │
│                                    └───────────┘        │                       │
│                                                         ▼                       │
│                ┌──────────────────────────────────────────────────────┐          │
│                │              Vector Store Layer                      │          │
│                │  (FAISS / Chroma / Qdrant / Pinecone / In-Memory)   │          │
│                └──────────────────────────────────────────────────────┘          │
│                                                                                 │
│  ┌──────────┐  ┌───────────────┐  ┌───────────┐  ┌──────────────────┐          │
│  │  Cache    │  │  Persistence  │  │ Embedding │  │  LLM Provider    │          │
│  │  Layer    │  │  Layer        │  │ Provider  │  │  Abstraction     │          │
│  └──────────┘  └───────────────┘  └───────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Data Flow — Two Modes

### Mode A: Hierarchical (Category → Tool)

```
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │  Embed Query     │
                    └────────┬────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Category Vector Space │
                 │  (Retrieve top-k      │
                 │   categories, k=2-3)  │
                 └───────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Category │  │ Category │  │ Category │
        │ A Tools  │  │ B Tools  │  │ C Tools  │
        │ (Vector  │  │ (Vector  │  │ (Vector  │
        │  Space)  │  │  Space)  │  │  Space)  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Merged Top-N    │
                  │  Candidate Tools │
                  │  (N = 3-5)       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  LLM Selector    │
                  │  (Final pick)    │  ◄── Optional
                  └────────┬─────────┘
                           │
                           ▼
                    Selected Tool(s)
```

### Mode B: Flat (Single Vector Space)

```
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │  Embed Query     │
                    └────────┬────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Global Tool Vector   │
                 │  Space (Retrieve      │
                 │  top-k tools, k=3-5)  │
                 └───────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │  LLM Selector    │
                  │  (Final pick)    │  ◄── Optional
                  └────────┬─────────┘
                           │
                           ▼
                    Selected Tool(s)
```

---

## 3. Component-by-Component Deep Dive

### 3.1 Tool Definition & Registry

The atomic unit. Every tool that enters ToolPickr gets normalized into an internal schema.

```
ToolDefinition:
  ├── name: str                          # function name
  ├── description: str                   # natural language description
  ├── parameters: dict (JSON Schema)     # input parameters schema
  ├── returns: str | dict                # return type / description
  ├── metadata: dict                     # arbitrary user metadata
  ├── category: Optional[str]            # user-assigned category (manual mode)
  └── tags: List[str]                    # optional tags for filtering
```

**ToolRegistry** is the central store:
- Register individual tools or bulk-register
- Support ingestion from: raw dicts, OpenAI function schemas, LangChain `Tool` objects, Python callables with docstrings (auto-extract via `inspect`)
- Deduplication by tool name
- CRUD operations on tools (add, remove, update, get)
- Event hooks: `on_tool_added`, `on_tool_removed` → triggers re-indexing

---

### 3.2 Embedding Engine

**Responsibility**: Convert tool metadata and user queries into dense vectors.

```
EmbeddingProvider (Abstract)
  ├── OpenAIEmbeddings        (text-embedding-3-small, text-embedding-3-large, ada-002)
  ├── CohereEmbeddings        (embed-english-v3.0, embed-multilingual-v3.0)
  ├── HuggingFaceEmbeddings   (sentence-transformers, local models)
  ├── VoyageEmbeddings        (voyage-3, etc.)
  ├── OllamaEmbeddings        (local Ollama models)
  └── CustomEmbeddings        (user-provided callable)
```

**Tool Text Rendering** — Before embedding, each tool is rendered into a text representation using a configurable template:

```
Default Template:
  "Tool: {name}
   Description: {description}
   Parameters: {formatted_params}
   Returns: {returns}"
```

Users can provide custom Jinja2-style templates or a callable `(ToolDefinition) → str`.

**Key Design Decisions:**
- All providers implement both `embed_text(str) → List[float]` and `embed_batch(List[str]) → List[List[float]]`
- Async variants: `aembed_text`, `aembed_batch`
- Dimension tracking: each provider reports its output dimensionality
- Normalization: optional L2 normalization for cosine similarity

---

### 3.3 Categorization Engine

Three strategies, one interface:

```
Categorizer (Abstract)
  │
  ├── ManualCategorizer
  │     User provides: Dict[str, List[str]]  →  { "email": ["send_email", "read_inbox"], ... }
  │
  ├── LLMCategorizer
  │     Sends all tool names + descriptions to an LLM with a prompt:
  │     "Group these N tools into logical categories..."
  │     Returns structured JSON grouping
  │     Supports chunking for very large tool sets (batch LLM calls)
  │
  ├── SemanticCategorizer
  │     1. Embed all tools
  │     2. Run clustering algorithm:
  │        - KMeans (when n_clusters known or estimated)
  │        - HDBSCAN (automatic cluster count)
  │        - Agglomerative (hierarchical, with distance threshold)
  │     3. Optionally use LLM to NAME the clusters
  │     4. Return groupings
  │
  └── HybridCategorizer
        SemanticCategorizer for initial grouping → LLM for refinement/naming
```

**Category Output Structure:**
```
Category:
  ├── name: str                      # "Email & Communication"
  ├── description: str               # Auto-generated or user-provided
  ├── tool_names: List[str]          # tools in this category
  └── centroid_embedding: List[float] # average embedding OR description embedding
```

**Auto Cluster Count Estimation:**
- Silhouette analysis
- Elbow method
- Or HDBSCAN's automatic detection
- Configurable: user can force a specific count

---

### 3.4 Vector Store Layer

```
VectorStore (Abstract)
  │
  ├── InMemoryStore        (NumPy-based, cosine/dot/euclidean)
  ├── FAISSStore           (faiss-cpu or faiss-gpu, IndexFlatIP/IndexIVFFlat/IndexHNSW)
  ├── ChromaStore          (ChromaDB collection wrapper)
  ├── QdrantStore          (Qdrant client wrapper)
  ├── PineconeStore        (Pinecone index wrapper)
  └── CustomStore          (user-provided implementation)
```

**Interface:**
```
- add(ids, embeddings, metadata)
- search(query_embedding, top_k, filters) → List[SearchResult]
- delete(ids)
- update(ids, embeddings, metadata)
- count() → int
- clear()
- persist() / load()
```

**Hierarchical mode uses multiple collections/indexes:**
- One "category index" storing category centroid/description embeddings
- N "tool indexes", one per category

**Flat mode uses one index:**
- Single "global tool index" with all tool embeddings

**Namespace strategy for hierarchical:**
```
vectorstore/
  ├── __category_index__
  ├── category_email_communication/
  ├── category_file_operations/
  ├── category_data_analysis/
  └── ...
```

---

### 3.5 Retrieval Engine

The orchestrator that ties embedding + vector store together.

```
Retriever (Abstract)
  │
  ├── FlatRetriever
  │     query → embed → search global index → top-k tools
  │
  └── HierarchicalRetriever
        query → embed →
          search category index → top-c categories →
            for each category: search category tool index → top-t tools →
              merge + deduplicate + rank → top-k tools
```

**Configurable Parameters:**
| Parameter | Description | Default |
|---|---|---|
| `category_top_k` | Categories to retrieve (hierarchical) | 3 |
| `tool_top_k` | Final tools to return | 5 |
| `tools_per_category` | Tools retrieved per category | 3 |
| `similarity_threshold` | Minimum score cutoff | 0.0 |
| `include_scores` | Return similarity scores | True |

**Re-Ranking (Optional):**
- After initial retrieval, optionally re-rank using:
  - Cross-encoder models (more accurate but slower)
  - Reciprocal Rank Fusion (when merging from multiple categories)
  - LLM-based re-ranking
  - Cohere Rerank API

```
ReRanker (Abstract)
  ├── CrossEncoderReRanker
  ├── RRFReRanker
  ├── CohereReRanker
  └── LLMReRanker
```

---

### 3.6 LLM Selector (Final Selection)

After retrieval narrows the field to 3-5 tools, optionally pass them to an LLM for final selection.

```
LLMSelector:
  Input:  user_query + List[ToolDefinition] (candidates)
  Output: List[ToolDefinition] (selected, usually 1-2) + reasoning

  Prompt Strategy:
    "Given the user's intent: '{query}'
     And these candidate tools:
     {tool_schemas}
     Select the best tool(s) and explain why."

  Structured output via JSON mode or function calling.
```

**LLM Provider Abstraction:**
```
LLMProvider (Abstract)
  ├── OpenAILLM           (gpt-4o, gpt-4o-mini, etc.)
  ├── AnthropicLLM        (claude-sonnet, etc.)
  ├── OllamaLLM           (local models)
  ├── LiteLLMLLM          (universal — 100+ providers via litellm)
  └── CustomLLM           (user-provided callable)
```

Interface:
```
- complete(prompt, **kwargs) → str
- acomplete(prompt, **kwargs) → str
- complete_structured(prompt, schema, **kwargs) → dict
```

---

### 3.7 Configuration Manager

Supports multiple configuration sources with precedence:

```
Priority (high → low):
  1. Explicit constructor arguments
  2. Environment variables (TOOLPICKR_*)
  3. Config file (toolpickr.yaml / toolpickr.json)
  4. Defaults
```

**Configuration Schema (Pydantic Settings):**
```yaml
toolpickr:
  mode: "hierarchical"          # "hierarchical" | "flat"

  embedding:
    provider: "openai"          # "openai" | "cohere" | "huggingface" | "ollama" | "voyage"
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimensions: null             # optional dimension reduction
    batch_size: 100
    tool_text_template: null     # custom Jinja2 template

  vectorstore:
    provider: "faiss"           # "memory" | "faiss" | "chroma" | "qdrant" | "pinecone"
    params: {}                  # provider-specific params

  categorization:
    method: "semantic"          # "manual" | "llm" | "semantic" | "hybrid" | "none"
    n_categories: "auto"        # int or "auto"
    algorithm: "kmeans"         # "kmeans" | "hdbscan" | "agglomerative"
    manual_categories: null     # Dict[str, List[str]] for manual mode

  retrieval:
    category_top_k: 3
    tool_top_k: 5
    tools_per_category: 3
    similarity_threshold: 0.0
    reranker: null              # "cross_encoder" | "rrf" | "cohere" | null

  selection:
    enabled: true
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    max_tools_to_select: 1

  llm:                          # LLM used for categorization / naming
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"

  cache:
    enabled: true
    backend: "memory"           # "memory" | "redis"
    ttl: 3600
    redis_url: null

  persistence:
    enabled: false
    backend: "filesystem"       # "filesystem" | "sqlite"
    path: "./toolpickr_data"

  logging:
    level: "INFO"
    format: "structured"        # "structured" | "plain"
```

---

### 3.8 Cache Layer

```
Cache (Abstract)
  ├── InMemoryCache     (dict + LRU eviction, bounded size)
  └── RedisCache        (Redis-backed, TTL support)
```

**What gets cached:**
| Key | Value | TTL |
|---|---|---|
| `embed:{hash(text)}` | embedding vector | long (24h) |
| `retrieve:{hash(query)}:{mode}` | retrieval results | short (5min) |
| `categorize:{hash(tool_set)}` | categorization results | long (until tools change) |

Cache invalidation: automatic on tool add/remove/update.

---

### 3.9 Persistence Layer

```
Persistence (Abstract)
  ├── FileSystemPersistence
  │     Saves: tool_registry.json, categories.json, embeddings.npz, vectorstore index files
  │     Location: configurable directory
  │
  └── SQLitePersistence
        Single .db file with tables for tools, categories, embeddings
```

**Enables:**
- `pickr.save(path)` / `pickr.load(path)` — snapshot entire state
- Avoid re-embedding and re-categorizing on restart
- Version tracking: schema version in metadata

---

### 3.10 Integration Adapters

```
integrations/
  ├── langchain.py
  │     - ToolPickrToolRetriever(BaseRetriever)  — LangChain retriever interface
  │     - as_langchain_tools(results) — convert to LangChain Tool objects
  │     - ToolPickrToolNode — for use in LangChain agents
  │
  ├── langgraph.py
  │     - toolpickr_node() — returns a LangGraph-compatible node function
  │     - Integrates into StateGraph as a tool-routing node
  │
  ├── openai_fc.py
  │     - to_openai_tools(results) — convert to OpenAI function calling format
  │     - ToolPickrOpenAIWrapper — drop-in for OpenAI's tools parameter
  │
  ├── crewai.py
  │     - ToolPickrCrewAITool — CrewAI-compatible tool wrapper
  │
  └── base.py
        - Abstract adapter interface for custom framework integrations
```

---

## 4. Internal Build Pipeline

When `pickr.build()` is called:

```
Step 1: Validate
  └── Validate all registered tools, check for duplicates, validate schemas

Step 2: Render
  └── Convert each ToolDefinition → text string using template

Step 3: Embed
  └── Batch-embed all tool text representations
  └── Cache embeddings

Step 4: Categorize (if mode == hierarchical)
  ├── If method == "manual": use user-provided mapping
  ├── If method == "semantic": cluster embeddings
  ├── If method == "llm": send to LLM for grouping
  └── If method == "hybrid": cluster then LLM-refine

Step 5: Build Vector Indexes
  ├── If hierarchical:
  │     ├── Compute category embeddings (centroids or embed descriptions)
  │     ├── Build category index
  │     └── For each category: build per-category tool index
  └── If flat:
        └── Build single global tool index

Step 6: Persist (if enabled)
  └── Save everything to disk

Step 7: Ready
  └── Mark pickr as ready, enable retrieval
```

---

## 5. Public API Surface

```python
# ─── Initialization ───
pickr = ToolPickr(config)
pickr = ToolPickr.from_yaml("config.yaml")
pickr = ToolPickr.from_dict({...})
pickr = ToolPickr(
    mode="hierarchical",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    ...
)

# ─── Tool Registration ───
pickr.register_tool(Tool(name=..., description=..., parameters=...))
pickr.register_tools([tool1, tool2, ...])
pickr.register_from_openai_schema(openai_tool_dict)
pickr.register_from_callable(my_function)          # auto-extract from docstring/signature
pickr.register_from_langchain_tool(lc_tool)
pickr.remove_tool("tool_name")
pickr.list_tools() → List[ToolDefinition]

# ─── Build ───
pickr.build()                    # full build pipeline
pickr.rebuild()                  # force full rebuild
pickr.add_tool_incremental(tool) # add without full rebuild (re-index incrementally)

# ─── Retrieval ───
results = pickr.retrieve("user query", top_k=3)
results = await pickr.aretrieve("user query", top_k=3)
# Returns: List[RetrievalResult(tool, score, category)]

# ─── Selection (retrieval + LLM pick) ───
selected = pickr.select("user query")
selected = await pickr.aselect("user query")
# Returns: SelectionResult(selected_tools, reasoning, candidates)

# ─── Output Formats ───
pickr.retrieve_as_openai_tools("query")     → List[dict]    # OpenAI function format
pickr.retrieve_as_langchain_tools("query")  → List[Tool]    # LangChain tools
pickr.retrieve_as_json("query")             → str           # JSON string

# ─── Introspection ───
pickr.categories → List[Category]
pickr.get_category("name") → Category
pickr.get_tool("name") → ToolDefinition
pickr.stats() → dict   # tool count, category count, index sizes, etc.

# ─── Persistence ───
pickr.save("./snapshot")
pickr = ToolPickr.load("./snapshot")

# ─── Export ───
pickr.export_categories_report() → str    # human-readable
pickr.export_tool_map() → dict            # { category: [tools] }
```

---

## 6. Project Structure

```
toolpickr/
│
├── pyproject.toml                        # Build config, dependencies, extras
├── README.md
├── CHANGELOG.md                          # skip for now
├── Makefile                              # dev commands (lint, format, build)
├── .env.example
│
├── docs/
│   ├── getting-started.md
│   ├── configuration.md
│   ├── architecture.md
│   ├── integrations.md
│   └── api-reference.md
│
├── examples/
│   ├── quickstart.py
│   ├── hierarchical_mode.py
│   ├── flat_mode.py
│   ├── custom_embeddings.py
│   ├── langchain_integration.py
│   ├── langgraph_integration.py
│   └── large_scale_200_tools.py
│
└── toolpickr/                            # ─── MAIN PACKAGE ───
    │
    ├── __init__.py                       # Public exports
    ├── _version.py                       # Version string
    ├── exceptions.py                     # All custom exceptions
    ├── types.py                          # Shared type aliases, enums, TypedDicts
    │
    ├── core/                             # ─── CORE DOMAIN ───
    │   ├── __init__.py
    │   ├── tool.py                       # ToolDefinition (Pydantic model)
    │   ├── category.py                   # Category (Pydantic model)
    │   ├── registry.py                   # ToolRegistry (add/remove/get/list)
    │   ├── config.py                     # ToolPickrConfig (Pydantic Settings)
    │   └── results.py                    # RetrievalResult, SelectionResult models
    │
    ├── embeddings/                       # ─── EMBEDDING PROVIDERS ───
    │   ├── __init__.py
    │   ├── base.py                       # EmbeddingProvider ABC
    │   ├── openai.py                     # OpenAIEmbeddings
    │   ├── cohere.py                     # CohereEmbeddings
    │   ├── huggingface.py                # HuggingFaceEmbeddings (sentence-transformers)
    │   ├── ollama.py                     # OllamaEmbeddings
    │   ├── voyage.py                     # VoyageEmbeddings
    │   ├── custom.py                     # CallableEmbeddings (user provides function)
    │   ├── renderer.py                   # ToolTextRenderer (template-based text generation)
    │   └── factory.py                    # get_embedding_provider(config) → provider
    │
    ├── vectorstores/                     # ─── VECTOR STORES ───
    │   ├── __init__.py
    │   ├── base.py                       # VectorStore ABC, SearchResult model
    │   ├── memory.py                     # InMemoryVectorStore (NumPy)
    │   ├── faiss.py                      # FAISSVectorStore
    │   ├── chroma.py                     # ChromaVectorStore
    │   ├── qdrant.py                     # QdrantVectorStore
    │   ├── pinecone.py                   # PineconeVectorStore
    │   └── factory.py                    # get_vectorstore(config) → store
    │
    ├── categorization/                   # ─── CATEGORIZATION ───
    │   ├── __init__.py
    │   ├── base.py                       # Categorizer ABC
    │   ├── manual.py                     # ManualCategorizer
    │   ├── semantic.py                   # SemanticCategorizer (clustering)
    │   ├── llm_categorizer.py            # LLMCategorizer
    │   ├── hybrid.py                     # HybridCategorizer (semantic + LLM naming)
    │   ├── cluster_utils.py              # Clustering helpers, optimal-k estimation
    │   └── factory.py                    # get_categorizer(config) → categorizer
    │
    ├── retrieval/                        # ─── RETRIEVAL ENGINE ───
    │   ├── __init__.py
    │   ├── base.py                       # Retriever ABC
    │   ├── flat.py                       # FlatRetriever
    │   ├── hierarchical.py               # HierarchicalRetriever
    │   ├── rerankers/
    │   │   ├── __init__.py
    │   │   ├── base.py                   # ReRanker ABC
    │   │   ├── rrf.py                    # Reciprocal Rank Fusion
    │   │   ├── cross_encoder.py          # Cross-encoder re-ranking
    │   │   └── cohere.py                 # Cohere Rerank API
    │   └── factory.py                    # get_retriever(config) → retriever
    │
    ├── selection/                        # ─── LLM SELECTION ───
    │   ├── __init__.py
    │   ├── base.py                       # Selector ABC
    │   ├── llm_selector.py               # LLMSelector implementation
    │   └── prompts.py                    # Selection prompt templates
    │
    ├── llm/                              # ─── LLM PROVIDER ABSTRACTION ───
    │   ├── __init__.py
    │   ├── base.py                       # LLMProvider ABC
    │   ├── openai.py                     # OpenAI chat completions
    │   ├── anthropic.py                  # Anthropic messages
    │   ├── ollama.py                     # Ollama local
    │   ├── litellm.py                    # LiteLLM (universal)
    │   └── factory.py                    # get_llm_provider(config) → provider
    │
    ├── cache/                            # ─── CACHING ───
    │   ├── __init__.py
    │   ├── base.py                       # Cache ABC
    │   ├── memory.py                     # InMemoryCache (LRU)
    │   ├── redis.py                      # RedisCache
    │   └── noop.py                       # NoOpCache (disabled)
    │
    ├── persistence/                      # ─── PERSISTENCE ───
    │   ├── __init__.py
    │   ├── base.py                       # Persistence ABC
    │   ├── filesystem.py                 # FileSystemPersistence (JSON + NPZ)
    │   └── sqlite.py                     # SQLitePersistence
    │
    ├── integrations/                     # ─── FRAMEWORK INTEGRATIONS ───
    │   ├── __init__.py
    │   ├── langchain.py                  # LangChain retriever adapter
    │   ├── langgraph.py                  # LangGraph node adapter
    │   ├── openai_fc.py                  # OpenAI function calling format
    │   └── crewai.py                     # CrewAI adapter
    │
    ├── utils/                            # ─── UTILITIES ───
    │   ├── __init__.py
    │   ├── hashing.py                    # Deterministic hashing for cache keys
    │   ├── logging.py                    # Structured logger setup
    │   ├── validation.py                 # Input validators
    │   ├── async_utils.py                # Sync/async bridge utilities
    │   └── imports.py                    # Lazy import helpers, optional dep checks
    │
    └── pickr.py                          # ─── MAIN FACADE CLASS ───
                                          # ToolPickr: ties everything together
```

---

## 7. Dependency Strategy

### `pyproject.toml` — Dependency Groups

```toml
[project]
name = "toolpickr"
version = "0.1.0"
requires-python = ">=3.10"

# ── Minimal core: ALWAYS installed ──
dependencies = [
    "pydantic>=2.0,<3.0",          # schemas, config validation
    "numpy>=1.24,<2.0",            # vector math, in-memory store
    "jinja2>=3.1,<4.0",            # tool text templates
    "pyyaml>=6.0",                  # yaml config loading
    "python-dotenv>=1.0",           # .env file support
    "structlog>=23.0",              # structured logging
    "tenacity>=8.0",                # retry logic for API calls
    "tiktoken>=0.5",                # token counting
]

[project.optional-dependencies]

# ── Embedding providers ──
openai       = ["openai>=1.0,<2.0"]
cohere       = ["cohere>=5.0"]
huggingface  = ["sentence-transformers>=2.2", "torch>=2.0"]
voyage       = ["voyageai>=0.2"]
ollama       = ["ollama>=0.1"]

# ── Vector stores ──
faiss-cpu    = ["faiss-cpu>=1.7"]
faiss-gpu    = ["faiss-gpu>=1.7"]
chroma       = ["chromadb>=0.4"]
qdrant       = ["qdrant-client>=1.7"]
pinecone     = ["pinecone-client>=3.0"]

# ── LLM providers ──
anthropic    = ["anthropic>=0.25"]
litellm      = ["litellm>=1.30"]

# ── Clustering ──
clustering   = ["scikit-learn>=1.3", "hdbscan>=0.8"]

# ── Re-ranking ──
rerankers    = ["sentence-transformers>=2.2"]

# ── Caching ──
redis        = ["redis>=5.0"]

# ── Framework integrations ──
langchain    = ["langchain-core>=0.2"]
langgraph    = ["langgraph>=0.0.30"]
crewai       = ["crewai>=0.28"]

# ── Convenience bundles ──
recommended  = [
    "toolpickr[openai,faiss-cpu,clustering]"
]
all = [
    "toolpickr[openai,cohere,huggingface,voyage,ollama]",
    "toolpickr[faiss-cpu,chroma,qdrant,pinecone]",
    "toolpickr[anthropic,litellm]",
    "toolpickr[clustering,rerankers,redis]",
    "toolpickr[langchain,langgraph,crewai]",
]

# ── Development ──
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.3",
    "mypy>=1.8",
    "pre-commit>=3.0",
    "mkdocs-material>=9.0",
]
```

---

## 8. Key Architectural Patterns Used

| Pattern | Where | Why |
|---|---|---|
| **Facade** | `ToolPickr` (pickr.py) | Single entry point hides all internal complexity |
| **Strategy** | Embedding, VectorStore, Categorizer, LLM | Swap implementations without changing calling code |
| **Abstract Factory** | All `factory.py` modules | Create correct provider from config string |
| **Builder** | `ToolPickr.from_yaml()`, constructor kwargs | Flexible construction |
| **Observer** | Registry events → re-indexing | Auto-maintain index consistency |
| **Template Method** | Build pipeline in `pickr.build()` | Fixed steps, variable implementations |
| **Adapter** | Integrations module | Convert ToolPickr interface to framework-specific |
| **Proxy/Cache** | Cache layer wrapping embedding/retrieval | Transparent caching |

---

## 9. Concurrency & Async Model

```
┌─────────────────────────────────────────────────┐
│ Every I/O method has sync + async variants:     │
│                                                  │
│   pickr.retrieve()    /  pickr.aretrieve()      │
│   pickr.select()      /  pickr.aselect()        │
│   pickr.build()       /  pickr.abuild()         │
│                                                  │
│ Internally:                                      │
│   - Async is native (aiohttp, async SDK calls)  │
│   - Sync wraps async via asyncio.run() bridge   │
│   - Thread-safe: registry uses RLock            │
│   - Batch embedding uses asyncio.gather()       │
└─────────────────────────────────────────────────┘
```

---

## 10. Error Handling Strategy

```
toolpickr/exceptions.py:

ToolPickrError (base)
  ├── ConfigurationError          # invalid config, missing API keys
  ├── ToolRegistrationError       # duplicate tool, invalid schema
  ├── EmbeddingError              # embedding API failure
  │   └── EmbeddingDimensionMismatchError
  ├── VectorStoreError            # index build/search failure
  ├── CategorizationError         # clustering failure, LLM grouping failure
  ├── RetrievalError              # search failed
  ├── SelectionError              # LLM selector failed
  ├── PersistenceError            # save/load failure
  ├── IntegrationError            # framework adapter failure
  └── ProviderNotInstalledError   # optional dependency missing
```

All external API calls wrapped with `tenacity` retries (configurable max retries, exponential backoff).

---

## 11. Observability

```
Logging (structlog):
  - Every major operation logged: build, embed, categorize, retrieve, select
  - Includes timing, token counts, result counts
  - Configurable level per component
  - JSON output for production, pretty output for dev

Metrics (exposed via .stats()):
  - Total tools registered
  - Categories count
  - Index sizes
  - Average retrieval latency (running average)
  - Cache hit/miss ratio
  - Embedding API calls count
  - LLM API calls count

Callbacks (optional hook system):
  - on_retrieve(query, results, latency)
  - on_select(query, selected, candidates)
  - on_build_complete(stats)
  - Enables custom monitoring integration
```

---

## 12. Scaling Considerations

| Scale | Strategy |
|---|---|
| **< 50 tools** | InMemoryStore, flat mode may suffice |
| **50–500 tools** | FAISS + hierarchical mode, recommended sweet spot |
| **500–5000 tools** | FAISS HNSW index or Qdrant, hierarchical mandatory |
| **5000+ tools** | Managed vector DB (Pinecone/Qdrant Cloud), async batch embedding, persistent indexes |

---

1. **`core/`** — Tool, Category, Config, Registry, Results models
2. **`embeddings/`** — Base + OpenAI provider + renderer
3. **`vectorstores/`** — Base + InMemory store
4. **`retrieval/`** — Base + FlatRetriever
5. **`pickr.py`** — Facade (wires everything, flat mode working end-to-end)
6. **`categorization/`** — Semantic categorizer
7. **`retrieval/hierarchical.py`** — Hierarchical retriever
8. **`selection/`** — LLM selector
9. **`persistence/`** + **`cache/`**
10. **`integrations/`** — LangChain first
11. Additional providers (Cohere, FAISS, Chroma, etc.)

# Features Built:
1. Faiss Index caching (LRU - last 5 builds)
2. 