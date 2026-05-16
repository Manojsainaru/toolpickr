from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ToolCallResult:
    """
    The result returned by ToolPickr.handle_tool_call().

    For action="search":
        - action will be "search"
        - data will contain {"available_tools": [...]} with tool schemas
        - tool_name will be None

    For action="execute" (when auto_execute=True and handler exists):
        - action will be "execute"
        - data will contain the tool's return value
        - tool_name will be the name of the tool that was called

    For action="execute" (when auto_execute=False or no handler):
        - action will be "execute"
        - data will contain {"tool_name": ..., "tool_arguments": ...}
          for the user to handle execution themselves
        - tool_name will be the name of the tool
    """
    action: str   # Which action was performed: "search" or "execute"
    success: bool # Whether the operation succeeded
    data: Dict[str, Any] = field(default_factory=dict)  # Response payload — contents depend on the action and auto_execute mode
    tool_name: Optional[str] = None  # The tool name (populated for "execute" actions)
    error: Optional[str] = None  # Error message if success is False
    executed: bool = False  # Whether the tool was actually auto-executed (True) or just routed (False)
