from typing import Callable, List, Optional, Any, Dict

from toolpickr.core.tool import ToolDefinition, Parameters, Property
from toolpickr.core.registry import ToolRegistry
from toolpickr.core.results import ToolCallResult
from toolpickr.embeddings.base import EmbeddingProvider
from toolpickr.embeddings.renderer import render_tool_text
from toolpickr.embeddings.factory import create_embedding_provider, DEFAULT_EMBEDDING_MODEL
from toolpickr.vectorstores.faiss import FaissVectorStore
from toolpickr.retrieval.flat import FlatRetriever
from toolpickr.retrieval.bm25 import BM25Retriever
from toolpickr.retrieval.ensemble import EnsembleRetriever
from toolpickr.vectorstores.base import SearchResult
import hashlib
import json
import os

_LRU_CACHE_DIR = "cache/vectorstores/faiss"
_LRU_MAX = 5
_LRU_MANIFEST = "lru_order.json"

class ToolPickr:
    def __init__(
            self,
            embedding_model: str = DEFAULT_EMBEDDING_MODEL,
            embedding_api_key: Optional[str] = None,
            reranker_model: Optional[str] = None,
            reranker_api_key: Optional[str] = None,
            top_k: int = 5,
            top_p: int = 15,
            semantic_weight: float = 0.5,
            bm25_weight: float = 0.5,
            auto_execute: bool = False,
            debug: bool = False,
        ) -> None:
        """
        Initializes the ToolPickr pipeline.

        Args:
            embedding_model:   Embedding model in "provider/model" format.
                               Default: "sentence_transformers/all-MiniLM-L6-v2" (local, no API key).
                               Examples: "google/gemini-embedding-001", "openai/text-embedding-3-small".
            embedding_api_key: Optional API key for the embedding provider.
                               If not provided, auto-discovered from environment variables.
            reranker_model:    Optional reranker model in "provider/model" format.
                               If None, no reranking is applied (ensemble result is final).
                               Example: "cross_encoder/ms-marco-MiniLM-L-6-v2".
            reranker_api_key:  Optional API key for the reranker provider.
                               If not provided, auto-discovered from environment variables.
            top_k:             Number of final tools to return per query (default: 5).
            top_p:             Number of ensemble candidates before reranking (default: 15).
                               Only used when reranker_model is set.
            semantic_weight:   Weight for semantic (embedding) scores in ensemble (default: 0.5).
            bm25_weight:       Weight for BM25 (lexical) scores in ensemble (default: 0.5).
            auto_execute:      If True, auto-execute tools with registered handlers.
                               If False, act as a router (default).
            debug:             If True, print debug logs to console.
        """
        self.debug = debug
        self.auto_execute = auto_execute
        self.top_k = top_k
        self.top_p = top_p
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.registry = ToolRegistry(debug=self.debug)

        # Initialize embedding provider via factory
        self._log(f"Initializing embedding provider: {embedding_model}")
        self.embeddings = create_embedding_provider(embedding_model, api_key=embedding_api_key)

        # Initialize vector store and retrievers
        self.vector_store = FaissVectorStore(dimension=self.embeddings.dimension, debug=self.debug)
        self.semantic_retriever = FlatRetriever(self.embeddings, self.vector_store, debug=self.debug)
        self.bm25_retriever = BM25Retriever(debug=self.debug)
        self.ensemble_retriever = EnsembleRetriever(
            semantic_retriever=self.semantic_retriever,
            bm25_retriever=self.bm25_retriever,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
            debug=self.debug,
        )

        # Initialize reranker (optional)
        self.reranker = None
        if reranker_model:
            self._log(f"Initializing reranker: {reranker_model}")
            from toolpickr.retrieval.reranker import Reranker
            self.reranker = Reranker(
                model_string=reranker_model,
                api_key=reranker_api_key,
                debug=self.debug,
            )

        # Tool texts cache for reranker (populated during build)
        self._tool_texts: Dict[str, str] = {}

        self._is_built = False

    def _log(self, msg: str) -> None:
        """Internal helper for logging debug messages."""
        if self.debug:
            print(f"[ToolPickr - Debug] {msg}")

    # ──────────────────────────────────────────────
    # Tool Registration
    # ──────────────────────────────────────────────

    def register_tool(self, tool: ToolDefinition, handler: Optional[Callable] = None) -> None:
        """Registers a single tool with an optional callable handler.
        
        Args:
            tool:    The tool definition (schema).
            handler: Optional callable that implements the tool. Required for
                     auto-execution when auto_execute=True.
        """
        self._log(f"Registering tool: {tool.name}")
        self.registry.register_tool(tool, handler=handler)

    def register_tools(self, tools: List[ToolDefinition]) -> None:
        """Bulk-registers a list of tools (without handlers)."""
        self._log(f"Bulk-registering {len(tools)} tools.")
        for tool in tools:
            self.registry.register_tool(tool)

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        """Registers a callable handler for an already-registered tool.
        
        Args:
            tool_name: Name of the tool (must already be registered).
            handler:   Callable that implements the tool.
        """
        self.registry.register_handler(tool_name, handler)

    # ──────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────

    def build(self) -> None:
        """
        Embeds all registered tools and builds the search indexes.

        Builds both:
          - FAISS vector index (for semantic/embedding retrieval)
          - BM25 index (for lexical retrieval)
        """
        self._log("Starting build process...")
        tools = self.registry.get_all_tools()
        if not tools:
            raise ValueError("No tools registered. Call register_tool() or register_tools() first.")

        names = [tool.name for tool in tools]
        texts = [render_tool_text(tool) for tool in tools]

        # Cache tool texts for reranker use
        self._tool_texts = dict(zip(names, texts))

        # Calculate the hash of names and texts combined
        combined_str = str(names) + str(texts)
        combined_hash = hashlib.sha256(combined_str.encode()).hexdigest()

        self._log(f"Embedding {len(tools)} tools...")
        
        # LRU disk cache (for FAISS index only)
        os.makedirs(_LRU_CACHE_DIR, exist_ok=True)
        manifest_path = os.path.join(_LRU_CACHE_DIR, _LRU_MANIFEST)

        # Load current LRU order list from disk
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                lru_order: list = json.load(f)
        else:
            lru_order = []

        # Cache hit - move entry to end
        if combined_hash in lru_order:
            self._log("FAISS index already present. Loading from disk.")
            lru_order.remove(combined_hash)
            lru_order.append(combined_hash)
            with open(manifest_path, "w") as f:
                json.dump(lru_order, f)
            self.vector_store.load_local(_LRU_CACHE_DIR, combined_hash)
        else:
            # Cache miss
            while len(lru_order) >= _LRU_MAX:
                evicted = lru_order.pop(0)
                self._log(f"LRU eviction: removing cached index '{evicted[:8]}...'")
                for ext in (".faiss", "_mapping.json"):
                    victim = os.path.join(_LRU_CACHE_DIR, evicted + ext)
                    if os.path.exists(victim):
                        os.remove(victim)

            # Build and persist the new FAISS index
            vectors = self.embeddings.embed_batch(texts)
            self.vector_store.add_vectors(vectors=vectors, tool_names=names)
            self.vector_store.save_local(_LRU_CACHE_DIR, combined_hash)

            # Record the new entry as most-recently used and save manifest
            lru_order.append(combined_hash)
            with open(manifest_path, "w") as f:
                json.dump(lru_order, f)

        # Always build the BM25 index (lightweight, no caching needed)
        self._log("Building BM25 index...")
        self.bm25_retriever.build(tool_names=names, tool_texts=texts)

        self._is_built = True
        self._log("Build complete.")

    # ──────────────────────────────────────────────
    # Raw Retrieval
    # ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Returns raw SearchResult objects for a query using the full pipeline
        (ensemble + optional reranker).

        Args:
            query:  Natural language query string.
            top_k:  Number of results. Defaults to self.top_k.

        Returns:
            List of SearchResult objects.
        """
        top_k = top_k or self.top_k
        self._log(f"Executing retrieval for query: '{query}' (top_k={top_k})")
        if not self._is_built:
            raise RuntimeError("Call .build() before retrieving.")

        results = self._retrieve_with_pipeline(query, top_k)
        self._log(f"Retrieval found {len(results)} results.")
        return results

    def _retrieve_with_pipeline(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Runs the full retrieval pipeline:
          1. Ensemble (semantic + BM25) → top_p candidates
          2. [Optional] Reranker → top_k final results
        """
        if self.reranker:
            # With reranker: ensemble fetches top_p, reranker narrows to top_k
            self._log(f"Pipeline: Ensemble (top_p={self.top_p}) → Reranker (top_k={top_k})")
            candidates = self.ensemble_retriever.retrieve(query, top_k=self.top_p)
            results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                tool_texts=self._tool_texts,
                top_k=top_k,
            )
        else:
            # Without reranker: ensemble returns top_k directly
            self._log(f"Pipeline: Ensemble (top_k={top_k})")
            results = self.ensemble_retriever.retrieve(query, top_k=top_k)

        return results

    # ──────────────────────────────────────────────
    # System Prompt
    # ──────────────────────────────────────────────

    def get_system_prompt(self, existing_prompt: str = "") -> str:
        """
        Returns the complete system prompt with ToolPickr instructions merged in.
        Pass the user's existing system prompt to merge, or empty string for standalone.

        The returned prompt instructs the LLM on the two-step toolpickr protocol:
          Step 1: Call toolpickr(action="search", queries=[...]) to find available tools.
          Step 2: Call toolpickr(action="execute", tool_name=..., tool_arguments={...})
                  to execute each tool.

        Example:
            system_prompt = pickr.get_system_prompt("You are a helpful assistant.")
        """
        toolpickr_instructions = (
            "You have access to a special meta-tool called 'toolpickr'. "
            "Use it to discover and invoke the right tools for any task.\n\n"
            "Follow this protocol:\n"
            "1. SEARCH: First, call toolpickr with action=\"search\" and a list of "
            "natural language queries describing what you need to do. Each query should "
            "describe one sub-task (e.g., [\"send an email\", \"get the weather\"]). "
            "ToolPickr will return the available tools that match your needs.\n"
            "2. EXECUTE: Then, for each tool you want to use, call toolpickr with "
            "action=\"execute\", the tool_name (from the search results), and "
            "tool_arguments (a JSON object matching the tool's parameter schema). "
            "ToolPickr will execute the tool and return the result.\n\n"
            "Important rules:\n"
            "- Always SEARCH first before executing. Do not guess tool names.\n"
            "- You can make multiple execute calls if you need multiple tools.\n"
            "- The tool_arguments must match the parameter schema returned in the search results.\n"
            "- After receiving tool results, synthesize the information to respond to the user."
        )
        if existing_prompt:
            return f"{existing_prompt}\n\n{toolpickr_instructions}"
        return toolpickr_instructions

    # ──────────────────────────────────────────────
    # toolpickr Tool Schema
    # ──────────────────────────────────────────────

    def get_tool(self, format: str = "base") -> Any:
        """
        Returns the schema for the `toolpickr` meta-tool, formatted for the
        requested LLM provider. This is the single tool the user adds to their
        LLM API call.

        The toolpickr tool supports two actions:
          - action="search": Discover tools by providing natural language queries.
          - action="execute": Execute a discovered tool by name with arguments.

        Args:
            format: "base" (plain dict) or "gemini"

        Returns:
            A tool schema ready to pass in the LLM's tools= parameter.

        Example:
            toolpickr_tool = pickr.get_tool(format="gemini")
            config = GenerateContentConfig(
                system_instruction=pickr.get_system_prompt("You are helpful."),
                tools=[toolpickr_tool, *other_tools],
            )
        """
        toolpickr_def = ToolDefinition(
            name="toolpickr",
            description=(
                "A meta-tool for discovering and executing tools. "
                "Use action='search' with a list of queries to find available tools, "
                "then use action='execute' with a tool_name and tool_arguments to run a tool."
            ),
            parameters=Parameters(
                type="object",
                properties={
                    "action": Property(
                        type="string",
                        description="The action to perform: 'search' to find tools, "
                                    "'execute' to run a discovered tool."
                    ),
                    "queries": Property(
                        type="array",
                        description="(For action='search') Natural language descriptions of "
                                    "the sub-tasks you need tools for. "
                                    "e.g. ['send an email', 'calculate total price']",
                        items={"type": "string"}
                    ),
                    "tool_name": Property(
                        type="string",
                        description="(For action='execute') The name of the tool to execute, "
                                    "as returned by the search results."
                    ),
                    "tool_arguments": Property(
                        type="object",
                        description="(For action='execute') The arguments to pass to the tool, "
                                    "matching the tool's parameter schema from the search results."
                    ),
                },
                required=["action"]
            )
        )

        if format == "gemini":
            from toolpickr.integrations.gemini import to_gemini_format
            return to_gemini_format([toolpickr_def])[0]

        # Base format — plain dict matching standard JSON Schema
        params = (
            toolpickr_def.parameters.model_dump(exclude_none=True)
            if hasattr(toolpickr_def.parameters, "model_dump")
            else toolpickr_def.parameters.dict(exclude_none=True)
        )
        return {
            "name": toolpickr_def.name,
            "description": toolpickr_def.description,
            "parameters": params,
        }

    # ──────────────────────────────────────────────
    # Handle Tool Call — The Core Entry Point
    # ──────────────────────────────────────────────

    def handle_tool_call(
        self,
        args: Dict[str, Any],
        format: str = "base",
    ) -> ToolCallResult:
        """
        Processes a toolpickr function call from the LLM.

        This is the single entry point for handling all toolpickr interactions.
        The LLM calls the 'toolpickr' tool with an action, and this method
        routes to the appropriate handler.

        Args:
            args:    The arguments dict from the LLM's function call.
                     Must contain "action" ("search" or "execute").
            format:  "base" (plain dicts) or "gemini" — controls tool schema format
                     in search results.

        Returns:
            ToolCallResult with the response data.

        Example:
            # In your tool-handling loop:
            if function_call.name == "toolpickr":
                result = pickr.handle_tool_call(dict(function_call.args))
                # Feed result.data back to the LLM as the tool response
        """
        action = args.get("action", "")
        self._log(f"Handling toolpickr call: action='{action}'")

        if action == "search":
            return self._handle_search(args, format)
        elif action == "execute":
            return self._handle_execute(args)
        else:
            return ToolCallResult(
                action=action,
                success=False,
                error=f"Unknown action: '{action}'. Use 'search' or 'execute'.",
            )

    # ──────────────────────────────────────────────
    # Search Retrieval (internal)
    # ──────────────────────────────────────────────

    def _handle_search(
        self,
        args: Dict[str, Any],
        format: str = "base",
    ) -> ToolCallResult:
        """Handles the 'search' action: retrieves matching tools for the given queries."""
        queries = args.get("queries", [])
        if not queries:
            return ToolCallResult(
                action="search",
                success=False,
                error="No queries provided. Pass a 'queries' array with search descriptions.",
            )

        if not self._is_built:
            raise RuntimeError("Call .build() before handling tool calls.")

        self._log(f"Searching for tools with {len(queries)} queries")

        seen = set()
        tool_definitions: List[ToolDefinition] = []

        for query in queries:
            self._log(f"  Retrieving for query: '{query}'")
            results = self._retrieve_with_pipeline(query, top_k=self.top_k)
            for res in results:
                if res.tool_name not in seen:
                    self._log(f"    Found unique tool: {res.tool_name}")
                    seen.add(res.tool_name)
                    tool_def = self.registry.get_tool(res.tool_name)
                    if tool_def:
                        tool_definitions.append(tool_def)

        self._log(f"Search returning {len(tool_definitions)} total unique tools.")

        # Build the tool info list for the LLM response
        available_tools = []
        for tool in tool_definitions:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.parameters:
                params = (
                    tool.parameters.model_dump(exclude_none=True)
                    if hasattr(tool.parameters, "model_dump")
                    else tool.parameters.dict(exclude_none=True)
                )
                tool_info["parameters"] = params
            available_tools.append(tool_info)

        return ToolCallResult(
            action="search",
            success=True,
            data={"available_tools": available_tools},
        )

    def _handle_execute(self, args: Dict[str, Any]) -> ToolCallResult:
        """Handles the 'execute' action: validates and optionally executes the tool."""
        tool_name = args.get("tool_name", "")
        tool_arguments = args.get("tool_arguments", {})

        if not tool_name:
            return ToolCallResult(
                action="execute",
                success=False,
                error="No tool_name provided. Specify which tool to execute.",
            )

        # Validate the tool exists
        tool_def = self.registry.get_tool(tool_name)
        if not tool_def:
            return ToolCallResult(
                action="execute",
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry.",
            )

        self._log(f"Execute request for tool '{tool_name}' with args: {tool_arguments}")

        # Auto-execute if enabled and handler exists
        if self.auto_execute:
            handler = self.registry.get_handler(tool_name)
            if handler:
                self._log(f"Auto-executing tool '{tool_name}'...")
                try:
                    result = handler(**tool_arguments) if tool_arguments else handler()
                    return ToolCallResult(
                        action="execute",
                        success=True,
                        data={"result": result},
                        tool_name=tool_name,
                        executed=True,
                    )
                except Exception as e:
                    return ToolCallResult(
                        action="execute",
                        success=False,
                        data={},
                        tool_name=tool_name,
                        executed=True,
                        error=f"Tool execution failed: {str(e)}",
                    )
            else:
                self._log(f"No handler registered for '{tool_name}', falling back to routing.")

        # Router mode: return the call info for the user to handle
        return ToolCallResult(
            action="execute",
            success=True,
            data={
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
            },
            tool_name=tool_name,
            executed=False,
        )

    # ──────────────────────────────────────────────
    # Legacy / Convenience Aliases
    # ──────────────────────────────────────────────

    def execute_search_tool(
        self,
        queries: List[str],
        top_k_per_query: Optional[int] = None,
        format: str = "base",
    ) -> List[Any]:
        """
        Runs retrieval for a list of natural language queries and returns
        the matching tool schemas formatted for the target LLM provider.

        This is a convenience method that wraps _handle_search for direct use.

        Args:
            queries:          List of natural language task descriptions.
            top_k_per_query:  How many tools to retrieve per query. Defaults to self.top_k.
            format:           "base" (plain dicts) or "gemini"

        Returns:
            List of tool schemas ready to pass as tools= in the next LLM call.
        """
        top_k = top_k_per_query or self.top_k
        self._log(f"Executing search tool for {len(queries)} queries (format='{format}')")
        if not self._is_built:
            raise RuntimeError("Call .build() before executing tool search.")

        seen = set()
        tool_definitions: List[ToolDefinition] = []

        for query in queries:
            self._log(f"  Retrieving for query: '{query}'")
            results = self._retrieve_with_pipeline(query, top_k=top_k)
            for res in results:
                if res.tool_name not in seen:
                    self._log(f"    Found unique tool: {res.tool_name}")
                    seen.add(res.tool_name)
                    tool_def = self.registry.get_tool(res.tool_name)
                    if tool_def:
                        tool_definitions.append(tool_def)
                        
        self._log(f"Search tool returning {len(tool_definitions)} total unique tools.")

        if format == "gemini":
            from toolpickr.integrations.gemini import to_gemini_format
            return to_gemini_format(tool_definitions)

        # Base format
        schemas = []
        for tool in tool_definitions:
            params = (
                tool.parameters.model_dump(exclude_none=True)
                if hasattr(tool.parameters, "model_dump")
                else tool.parameters.dict(exclude_none=True)
            )
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            })
        return schemas
