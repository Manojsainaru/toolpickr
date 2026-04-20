from typing import List, Any
from toolpickr.core.tool import ToolDefinition


def to_gemini_format(tool_defs: List[ToolDefinition]) -> list[dict]:
    """Converts ToolPickr ToolDefinitions into the Google function_declarations format.
    
    The Google Gen AI SDK expects: [{"function_declarations": [...]}]
    """
    declarations = []
    
    for tool in tool_defs:
        declaration = {
            "name": tool.name,
            "description": tool.description,
        }
        
        if tool.parameters:
            # Use model_dump (Pydantic v2) to safely serialize nested models.
            # exclude_none=True avoids sending empty keys like 'items' to the API.
            if hasattr(tool.parameters, "model_dump"):
                declaration["parameters"] = tool.parameters.model_dump(exclude_none=True)
            else:
                declaration["parameters"] = tool.parameters.dict(exclude_none=True)
                
        declarations.append(declaration)
        
    return [{"function_declarations": declarations}]


def build_tool_result_message(original_model_content: Any, retrieved_tool_names: List[str]) -> Any:
    """
    Builds the two conversation turns needed after a tool_search call:
      1. The model turn — the original response content, passed as-is to
         preserve any thought_signature required by Gemini thinking models.
      2. The tool turn  — ToolPickr's response listing the retrieved tools.

    Returns a tuple: (model_content, tool_content)
    These are ready to be appended to the developer's conversation history.
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai is required. Install with: pip install google-genai")

    # Turn 1: use the original model content directly so that thought_signature
    # and all other parts (thoughts, etc.) are preserved exactly as received.
    model_turn = original_model_content

    # Turn 2: ToolPickr's tool response — tells Gemini which tools are now available
    tool_message = (
        f"The following tools are now available to complete the task: "
        f"{retrieved_tool_names}. Use them to fulfill the user's request."
    )
    tool_turn = types.Content(
        role="tool",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="tool_search",
                    response={"result": tool_message}
                )
            )
        ]
    )

    return model_turn, tool_turn


# ──────────────────────────────────────────────────────────────────────
# Wrapped Client  — transparent two-turn orchestration
# ──────────────────────────────────────────────────────────────────────

class WrappedResponse:
    """
    Wraps the native Gemini response with ToolPickr metadata.

    Proxies all attribute access to the underlying response, so developers
    can use it exactly like a native response object (e.g. response.candidates,
    response.text, etc.) while also inspecting ToolPickr-specific fields:

        response.toolpickr_intercepted         → bool
        response.toolpickr_retrieved_tool_names → List[str]
        response.toolpickr_history              → full conversation history
    """

    def __init__(self, response, intercepted=False, retrieved_tool_names=None, history=None):
        self._response = response
        self.toolpickr_intercepted = intercepted
        self.toolpickr_retrieved_tool_names = retrieved_tool_names or []
        self.toolpickr_history = history or []

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __repr__(self):
        return (
            f"WrappedResponse(intercepted={self.toolpickr_intercepted}, "
            f"retrieved={self.toolpickr_retrieved_tool_names})"
        )


class ModelsProxy:
    """Proxies the 'models' attribute of the google.genai.Client to intercept generate_content."""
    
    def __init__(self, models, pickr, format):
        self._models = models
        self._pickr = pickr
        self._format = format

    def generate_content(self, *, model: str, contents: Any, config: Any = None, **kwargs) -> WrappedResponse:
        """Intercepts the native models.generate_content call."""
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai is required. Install with: pip install google-genai")

        is_dict_config = isinstance(config, dict)
        if config is None:
            config = types.GenerateContentConfig()

        # Extract system_instruction and tools
        system_instruction = ""
        extra_tools = []
        if is_dict_config:
            system_instruction = config.get("system_instruction", "")
            extra_tools = config.get("tools", [])
        else:
            system_instruction = getattr(config, "system_instruction", "")
            if system_instruction is None:
                system_instruction = ""
            extra_tools = getattr(config, "tools", [])

        if extra_tools is None:
            extra_tools = []

        # Merge system instructions
        final_system = self._pickr.inject_system_prompt(system_instruction)

        # Merge tools
        search_tool = self._pickr.get_search_tool(format=self._format)
        turn1_tools = [search_tool] + extra_tools

        # Create Turn 1 config
        turn1_config = self._clone_and_update_config(config, final_system, turn1_tools, types, is_dict_config)

        # Turn 1 generate call
        response = self._models.generate_content(
            model=model,
            contents=contents,
            config=turn1_config,
            **kwargs,
        )

        # Intercept logic
        intercept = self._pickr.handle_response(response, format=self._format)

        # Normalize contents to a list for history tracking
        contents_list = contents if isinstance(contents, list) else [contents]

        if not intercept.is_tool_search:
            # No tool_search triggered
            return WrappedResponse(
                response=response,
                intercepted=False,
                history=contents_list + [response.candidates[0].content],
            )

        # Turn 2: tool_search was triggered
        updated_contents = contents_list + intercept.updated_history
        turn2_tools = intercept.retrieved_tools + extra_tools
        turn2_config = self._clone_and_update_config(config, final_system, turn2_tools, types, is_dict_config)

        response2 = self._models.generate_content(
            model=model,
            contents=updated_contents,
            config=turn2_config,
            **kwargs,
        )

        # Collect retrieved names
        retrieved_names = [
            decl["name"]
            for group in intercept.retrieved_tools
            for decl in group.get("function_declarations", [])
        ]

        full_history = updated_contents + [response2.candidates[0].content]

        return WrappedResponse(
            response=response2,
            intercepted=True,
            retrieved_tool_names=retrieved_names,
            history=full_history,
        )

    def _clone_and_update_config(self, config, system_instruction, tools, types, is_dict_config):
        if is_dict_config:
            new_config = dict(config)
            new_config["system_instruction"] = system_instruction
            new_config["tools"] = tools
            return new_config
        else:
            kw = {}
            if hasattr(config, "model_dump"):
                # Pydantic v2
                kw = config.model_dump(exclude_unset=True)
            elif hasattr(config, "dict"):
                # Pydantic v1 fallback
                kw = config.dict(exclude_unset=True)
            else:
                kw = {k: v for k, v in vars(config).items() if not k.startswith('_')}
                
            kw["system_instruction"] = system_instruction
            kw["tools"] = tools
            return types.GenerateContentConfig(**kw)

    def __getattr__(self, name):
        return getattr(self._models, name)


class WrappedGeminiClient:
    """
    A transparent proxy around the Gemini genai.Client that auto-handles the
    tool_search two-turn orchestration without modifying native methods.

    Usage:
        smart_client = pickr.wrap(gemini_client, format="gemini")
        
        # Behaves exactly natively:
        response = smart_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
            )
        )
        
        # response is the FINAL response — tool_search was handled internally
    """

    def __init__(self, client, pickr, format="gemini"):
        self._client = client
        self._pickr = pickr
        self._format = format

    def __getattr__(self, name):
        """Proxy all accesses to the original client. Intercept `.models` specifically."""
        attr = getattr(self._client, name)
        if name == "models":
            return ModelsProxy(attr, self._pickr, self._format)
        return attr
