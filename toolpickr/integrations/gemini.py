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
