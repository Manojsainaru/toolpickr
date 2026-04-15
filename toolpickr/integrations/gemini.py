from typing import List, Any
from toolpickr.core.tool import ToolDefinition


def to_gemini_format(tool_defs: List[ToolDefinition]) -> list[dict]:
    """Converts ToolPickr ToolDefinitions into the Gemini function_declarations format.
    
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


def build_tool_result_message(function_call: Any, retrieved_tool_names: List[str]) -> Any:
    """
    Builds the two conversation turns needed after a tool_search call:
      1. The model turn — echoing what Gemini said (its tool_search call).
      2. The tool turn  — ToolPickr's response listing the retrieved tools.

    Returns a tuple: (model_content, tool_content)
    These are ready to be appended to the developer's conversation history.
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai is required. Install with: pip install google-genai")

    # Turn 1: echo Gemini's model response (the tool_search function call)
    model_turn = types.Content(
        role="model",
        parts=[types.Part(function_call=function_call)]
    )

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
