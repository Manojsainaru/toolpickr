from typing import List, Optional, Any

from toolpickr.core.tool import ToolDefinition, Parameters, Property
from toolpickr.core.registry import ToolRegistry
from toolpickr.core.results import InterceptResult
from toolpickr.embeddings.base import EmbeddingProvider
from toolpickr.embeddings.renderer import render_tool_text
from toolpickr.vectorstores.faiss import FaissVectorStore
from toolpickr.retrieval.flat import FlatRetriever
from toolpickr.vectorstores.base import SearchResult


class ToolPickr:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store=None, debug: bool = False):
        """
        Initializes the ToolPickr pipeline.

        Args:
            embedding_provider: Any EmbeddingProvider instance (GeminiEmbeddings,
                                HuggingFaceEmbeddings, CohereEmbeddings, etc.)
            vector_store:       Optional. A VectorStore instance. Defaults to a
                                FaissVectorStore auto-sized to the provider's dimension.
            debug:              Optional. If True, prints execution steps and logs to console.
        """
        self.debug = debug
        self.registry = ToolRegistry(debug=self.debug)
        self.embeddings = embedding_provider
        self.vector_store = vector_store or FaissVectorStore(dimension=self.embeddings.dimension, debug=self.debug)
        if vector_store and hasattr(self.vector_store, 'debug'):
            self.vector_store.debug = self.debug
        self.retriever = FlatRetriever(self.embeddings, self.vector_store, debug=self.debug)
        self._is_built = False

    def _log(self, msg: str) -> None:
        """Internal helper for logging debug messages."""
        if self.debug:
            print(f"[ToolPickr - Debug] {msg}")

    # ──────────────────────────────────────────────
    # Tool Registration
    # ──────────────────────────────────────────────

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

        self._log(f"Embedding {len(tools)} tools...")
        vectors = self.embeddings.embed_batch(texts)

        self.vector_store.add_vectors(vectors=vectors, tool_names=names)
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
                tools=[pickr.get_search_tool_schema(format="gemini")]
            )
        """
        toolpickr_instructions = self.get_system_instructions()
        if existing_prompt:
            return f"{existing_prompt}\n\n{toolpickr_instructions}"
        return toolpickr_instructions

    # ──────────────────────────────────────────────
    # tool_search Schema
    # ──────────────────────────────────────────────

    def get_search_tool_schema(self, format: str = "base") -> Any:
        """
        Returns the schema for the `tool_search` function, formatted for the
        requested LLM provider.

        Args:
            format: "base" (plain dict) or "gemini"

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
            format:        "base" or "gemini" — controls schema output format.

        Returns:
            InterceptResult

        Example (Gemini):
            response = client.models.generate_content(...)
            intercept = pickr.handle_response(response, format="gemini")

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

        # Build the conversation turns for the developer's history
        updated_history = self._build_history_turns(function_call, retrieved_tools, format)

        return InterceptResult(
            is_tool_search=True,
            retrieved_tools=retrieved_tools,
            updated_history=updated_history,
            original_function_call=function_call,
        )

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    def _extract_function_call(self, llm_response: Any) -> Optional[Any]:
        """
        Attempts to extract a function_call from the LLM response.
        Handles Gemini SDK response shapes. Returns None if not found.
        """
        try:
            # Gemini SDK: response.candidates[0].content.parts[0].function_call
            part = llm_response.candidates[0].content.parts[0]
            fc = getattr(part, "function_call", None)
            return fc if fc is not None else None
        except (AttributeError, IndexError, TypeError):
            return None

    def _build_history_turns(
        self,
        function_call: Any,
        retrieved_tools: List[Any],
        format: str,
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
            model_turn, tool_turn = build_tool_result_message(function_call, retrieved_names)
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
