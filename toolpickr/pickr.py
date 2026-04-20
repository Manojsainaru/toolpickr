from typing import List, Optional, Any

from toolpickr.core.tool import ToolDefinition, Parameters, Property
from toolpickr.core.registry import ToolRegistry
from toolpickr.core.results import InterceptResult
from toolpickr.embeddings.base import EmbeddingProvider
from toolpickr.embeddings.renderer import render_tool_text
from toolpickr.vectorstores.faiss import FaissVectorStore
from toolpickr.retrieval.flat import FlatRetriever
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
            embedding_provider: str = "",
            embedding_model: str = "",
            embedding_api_key: str = "",
            debug: bool = False,
        ) -> None:
        """
        Initializes the ToolPickr pipeline.

        Args:
            embedding_provider: Any Embedding provider: "google", "openai", etc
            debug:              Optional. If True, prints execution steps and logs to console.
        """
        self.debug = debug
        self.registry = ToolRegistry(debug=self.debug)
        self.embeddings = self.get_embedding_provider(embedding_provider, embedding_model, embedding_api_key)
        self.vector_store = FaissVectorStore(dimension=self.embeddings.dimension, debug=self.debug)
        self.retriever = FlatRetriever(self.embeddings, self.vector_store, debug=self.debug)
        self._is_built = False

    def _log(self, msg: str) -> None:
        """Internal helper for logging debug messages."""
        if self.debug:
            print(f"[ToolPickr - Debug] {msg}")

    # ──────────────────────────────────────────────
    # Tool Registration
    # ──────────────────────────────────────────────

    def get_embedding_provider(self, embedding_provider: str, embedding_model: str, embedding_api_key: str = None) -> EmbeddingProvider:
        if embedding_provider == "sentence_transformers":
            return None # TODO: sentence transformers embedding provider code to be implemented first
        elif embedding_provider == "google":
            from toolpickr.embeddings.google import GoogleEmbeddings
            return GoogleEmbeddings(api_key=embedding_api_key, model=embedding_model)
        else: 
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

    def register_tool(self, tool: ToolDefinition) -> None:
        """Registers a single tool."""
        self._log(f"Registering tool: {tool.name}")
        self.registry.register_tool(tool)

    def register_tools(self, tools: List[ToolDefinition]) -> None:
        """Bulk-registers a list of tools."""
        self._log(f"Bulk-registering {len(tools)} tools.")
        for tool in tools:
            self.registry.register_tool(tool)

    # ──────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────

    def build(self) -> None:
        """Embeds all registered tools and builds the search index."""
        self._log("Starting build process...")
        tools = self.registry.get_all_tools()
        if not tools:
            raise ValueError("No tools registered. Call register_tool() or register_tools() first.")

        names = [tool.name for tool in tools]
        texts = [render_tool_text(tool) for tool in tools]

        # Calculate the hash of names and texts combined
        combined_str = str(names) + str(texts)
        combined_hash = hashlib.sha256(combined_str.encode()).hexdigest()

        self._log(f"Embedding {len(tools)} tools...")
        
        # LRU disk cache
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
            self._log("Index file already present. Loading from disk.")
            lru_order.remove(combined_hash)
            lru_order.append(combined_hash)
            with open(manifest_path, "w") as f:
                json.dump(lru_order, f)
            self.vector_store.load_local(_LRU_CACHE_DIR, combined_hash)
            self._is_built = True
            self._log("Build complete.")
            return

        # Cache miss
        while len(lru_order) >= _LRU_MAX:
            evicted = lru_order.pop(0)
            self._log(f"LRU eviction: removing cached index '{evicted[:8]}...'")
            for ext in (".faiss", "_mapping.json"):
                victim = os.path.join(_LRU_CACHE_DIR, evicted + ext)
                if os.path.exists(victim):
                    os.remove(victim)

        # Build and persist the new index
        vectors = self.embeddings.embed_batch(texts)
        self.vector_store.add_vectors(vectors=vectors, tool_names=names)
        self.vector_store.save_local(_LRU_CACHE_DIR, combined_hash)

        # Record the new entry as most-recently used and save manifest
        lru_order.append(combined_hash)
        with open(manifest_path, "w") as f:
            json.dump(lru_order, f)

        self._is_built = True
        self._log("Build complete.")

    # ──────────────────────────────────────────────
    # Raw Retrieval
    # ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Returns raw SearchResult objects for a query. Useful for inspection."""
        self._log(f"Executing raw retrieval for query: '{query}' (top_k={top_k})")
        if not self._is_built:
            raise RuntimeError("Call .build() before retrieving.")
        results = self.retriever.retrieve(query, top_k)
        self._log(f"Raw retrieval found {len(results)} results.")
        return results

    # ──────────────────────────────────────────────
    # System Prompt Injection
    # ──────────────────────────────────────────────

    def get_system_instructions(self) -> str:
        """Returns ToolPickr's system instructions as a standalone string."""
        return (
            "You are an AI assistant capable of performing complex tasks using tools. "
            "You currently have access to 'tool_search'. "
            "When given a task, you MUST FIRST call 'tool_search' with a list of natural "
            "language queries that describe the sub-tasks you need to perform. "
            "The system will then provide the matching tools, which you should use to "
            "complete the task."
        )

    def inject_system_prompt(self, existing_prompt: str = "") -> str:
        """
        Appends ToolPickr's instructions to the developer's existing system prompt.
        Pass an empty string (default) if there is no existing prompt.

        Example:
            config = GenerateContentConfig(
                system_instruction=pickr.inject_system_prompt("You are a helpful assistant."),
                tools=[pickr.get_search_tool(format="google")]
            )
        """
        toolpickr_instructions = self.get_system_instructions()
        if existing_prompt:
            return f"{existing_prompt}\n\n{toolpickr_instructions}"
        return toolpickr_instructions

    # ──────────────────────────────────────────────
    # tool_search Schema
    # ──────────────────────────────────────────────

    def get_search_tool(self, format: str = "base") -> Any:
        """
        Returns the schema for the `tool_search` function, formatted for the
        requested LLM provider.

        Args:
            format: "base" (plain dict) or "google"

        Returns:
            A dict formatted for the target LLM provider.
        """
        search_tool_def = ToolDefinition(
            name="tool_search",
            description=(
                "Search for tools required to complete the user's task. "
                "Pass a list of natural language queries describing each action needed."
            ),
            parameters=Parameters(
                type="object",
                properties={
                    "queries": Property(
                        type="array",
                        description="Natural language descriptions of actions needed, "
                                    "e.g. ['send an email', 'calculate total price']",
                        items={"type": "string"}
                    )
                },
                required=["queries"]
            )
        )

        if format == "gemini":
            from toolpickr.integrations.gemini import to_gemini_format
            return to_gemini_format([search_tool_def])[0]

        # Base format — plain dict matching standard JSON Schema
        params = (
            search_tool_def.parameters.model_dump(exclude_none=True)
            if hasattr(search_tool_def.parameters, "model_dump")
            else search_tool_def.parameters.dict(exclude_none=True)
        )
        return {
            "name": search_tool_def.name,
            "description": search_tool_def.description,
            "parameters": params,
        }

    # ──────────────────────────────────────────────
    # tool_search Execution
    # ──────────────────────────────────────────────

    def execute_search_tool(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        format: str = "base",
    ) -> List[Any]:
        """
        Runs retrieval for a list of natural language queries and returns
        the matching tool schemas formatted for the target LLM provider.

        Args:
            queries:          List of natural language task descriptions.
            top_k_per_query:  How many tools to retrieve per query.
            format:           "base" (plain dicts) or "gemini"

        Returns:
            List of tool schemas ready to pass as tools= in the next LLM call.
        """
        self._log(f"Executing search tool for {len(queries)} queries (format='{format}')")
        if not self._is_built:
            raise RuntimeError("Call .build() before executing tool search.")

        seen = set()
        tool_definitions: List[ToolDefinition] = []

        for query in queries:
            self._log(f"  Retrieving for query: '{query}'")
            results = self.retriever.retrieve(query, top_k=top_k_per_query)
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

    # ──────────────────────────────────────────────
    # Response Interceptor  ← The magic method
    # ──────────────────────────────────────────────

    def handle_response(self, llm_response: Any, format: str = "base") -> InterceptResult:
        """
        Inspects an LLM response. If it contains a `tool_search` function call,
        ToolPickr automatically runs retrieval and builds the conversation turns
        required for the second LLM call.

        The developer only needs to check `result.is_tool_search` and, if True,
        use `result.updated_history` and `result.retrieved_tools` in the next call.

        Args:
            llm_response:  The raw response object from the LLM SDK.
            format:        "base" or "google" — controls schema output format.

        Returns:
            InterceptResult

        Example (Google):
            response = client.models.generate_content(...)
            intercept = pickr.handle_response(response, format="google")

            if intercept.is_tool_search:
                response2 = client.models.generate_content(
                    contents=my_history + intercept.updated_history,
                    config=GenerateContentConfig(tools=intercept.retrieved_tools)
                )
        """
        self._log("Inspecting LLM response for tool_search call...")
        function_call = self._extract_function_call(llm_response)

        # Not a tool_search call — pass through untouched
        if function_call is None or getattr(function_call, "name", None) != "tool_search":
            self._log("No tool_search function call found. Returning standard InterceptResult.")
            return InterceptResult(is_tool_search=False)

        self._log(f"tool_search call intercepted! Arguments: {getattr(function_call, 'args', {})}")

        # Extract the queries the LLM wants to search for
        queries = list(function_call.args.get("queries", []))

        # Run retrieval
        retrieved_tools = self.execute_search_tool(
            queries=queries,
            top_k_per_query=3,
            format=format,
        )

        # Original model content preserves thought_signature required by thinking models
        original_model_content = llm_response.candidates[0].content

        # Build the conversation turns for the developer's history
        updated_history = self._build_history_turns(function_call, retrieved_tools, format, original_model_content)

        return InterceptResult(
            is_tool_search=True,
            retrieved_tools=retrieved_tools,
            updated_history=updated_history,
            original_function_call=function_call,
        )

    # ──────────────────────────────────────────────
    # Wrapped Client
    # ──────────────────────────────────────────────

    def wrap(self, client: Any, format: str = "gemini") -> Any:
        """
        Wraps an LLM client so that tool_search interception happens
        transparently. The returned object acts as a native proxy, allowing
        developers to use the native methods exactly as before, while
        auto-handling the two-turn orchestration behind the scenes.

        The developer can still pass additional tools and configs through 
        the native parameters (e.g. `config` in Gemini) — full flexibility is preserved.

        Args:
            client: The native LLM client (e.g. genai.Client instance).
            format: The LLM provider format ("gemini", etc.)

        Returns:
            A transparent proxy around the native client.

        Example (Gemini):
            from google.genai import types
            
            smart_client = pickr.wrap(gemini_client, format="gemini")
            response = smart_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant."
                )
            )
            # response behaves natively but with metadata:
            # response.toolpickr_intercepted → bool
            # response.toolpickr_retrieved_tool_names → list[str]
        """
        self._log(f"Wrapping client with format='{format}'")
        if not self._is_built:
            raise RuntimeError("Call .build() before wrapping a client.")

        if format == "gemini":
            from toolpickr.integrations.gemini import WrappedGeminiClient
            return WrappedGeminiClient(client=client, pickr=self, format=format)

        raise ValueError(f"Unsupported format for wrap(): '{format}'. Supported: 'gemini'")

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    def _extract_function_call(self, llm_response: Any) -> Optional[Any]:
        """
        Attempts to extract a function_call from the LLM response.
        Iterates all parts to handle thinking models where parts[0] may be
        a thought and the function_call appears in a later part.
        """
        try:
            for part in llm_response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    return fc
            return None
        except (AttributeError, IndexError, TypeError):
            return None

    def _build_history_turns(
        self,
        function_call: Any,
        retrieved_tools: List[Any],
        format: str,
        original_model_content: Any = None,
    ) -> List[Any]:
        """
        Builds the two conversation turns (model + tool) that the developer
        needs to append to their history before the second LLM call.
        """
        if format == "gemini":
            from toolpickr.integrations.gemini import build_tool_result_message

            # Collect readable tool names for the tool response message
            retrieved_names = [
                decl["name"]
                for group in retrieved_tools
                for decl in group.get("function_declarations", [])
            ]
            # Pass original_model_content so thought_signature is preserved
            model_turn, tool_turn = build_tool_result_message(original_model_content, retrieved_names)
            return [model_turn, tool_turn]

        # Base format — return plain dicts the developer can inspect or adapt
        retrieved_names = [t.get("name", "") for t in retrieved_tools]
        return [
            {"role": "model", "function_call": {"name": "tool_search", "args": dict(function_call.args)}},
            {
                "role": "tool",
                "function_response": {
                    "name": "tool_search",
                    "response": {"result": f"Available tools: {retrieved_names}"}
                }
            }
        ]
