# Tool registry -mappings
from typing import Callable, Dict, List, Optional
from toolpickr.core.tool import ToolDefinition

class ToolRegistry:
    def __init__(self, debug: bool = False):
        self._tools: Dict[str, ToolDefinition] = {}     # Internal dict mapping the tool's name to its definition
        self._handlers: Dict[str, Callable] = {}        # Optional callable handlers for tools (tool_name -> callable)
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[ToolRegistry - Debug] {msg}")

    def register_tool(self, tool: ToolDefinition, handler: Optional[Callable] = None) -> None:
        """Adds a tool to the registry. Overwrites if a tool with the same name already exists.
        
        Args:
            tool:    The tool definition (schema).
            handler: Optional callable that implements the tool. If provided, ToolPickr
                     can auto-execute the tool when the LLM requests it.
        """
        if tool.name in self._tools:
            self._log(f"Overwriting existing tool: {tool.name}")
        else:
            self._log(f"Registering new tool: {tool.name}")
        self._tools[tool.name] = tool
        if handler is not None:
            self._handlers[tool.name] = handler
            self._log(f"Handler registered for tool: {tool.name}")

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        """Registers a callable handler for an already-registered tool.
        
        Args:
            tool_name: Name of the tool (must already be registered).
            handler:   Callable that implements the tool.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Cannot register handler: tool '{tool_name}' is not registered.")
        self._handlers[tool_name] = handler
        self._log(f"Handler registered for tool: {tool_name}")

    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """Returns the callable handler for a tool, or None if no handler is registered."""
        return self._handlers.get(tool_name)
        
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Fetches a tool by name. Returns None if it doesn't exist."""
        tool = self._tools.get(name)
        if not tool:
            self._log(f"Tool not found: {name}")
        return tool

    def get_all_tools(self) -> List[ToolDefinition]:
        """Returns a list of all registered tools."""
        return list(self._tools.values())
        
    def remove_tool(self, name: str) -> bool:
        """Removes a tool by name. Returns True if removed, False if it wasn't found."""
        if name in self._tools:
            del self._tools[name]
            self._handlers.pop(name, None)
            self._log(f"Removed tool: {name}")
            return True
        self._log(f"Failed to remove tool (not found): {name}")
        return False
        
    def __len__(self) -> int:
        """Allows you to run `len(registry)` to see how many tools are registered."""
        return len(self._tools)
