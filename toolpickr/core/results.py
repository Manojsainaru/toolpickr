from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class InterceptResult:
    """
    The result returned by ToolPickr.handle_response().

    If `is_tool_search` is True, the developer should:
      1. Append `updated_history` to their conversation.
      2. Make a second LLM call with `tools=retrieved_tools`.

    If `is_tool_search` is False, ToolPickr did not intercept anything.
    The developer handles the response as normal.
    """

    # Was this response a tool_search call that ToolPickr handled?
    is_tool_search: bool

    # The full retrieved tool schemas, ready to pass as tools= in the next LLM call.
    # Will be an empty list if is_tool_search is False.
    retrieved_tools: List[Any] = field(default_factory=list)

    # Pre-built conversation turns to append to the developer's history:
    #   [model_turn (the tool_search call), tool_turn (ToolPickr's response)]
    # Developer just does: history += intercept.updated_history
    updated_history: List[Any] = field(default_factory=list)

    # The raw function_call object from the LLM, exposed for advanced use cases.
    original_function_call: Any = None
