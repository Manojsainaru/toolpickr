# Tool registry -mappings
from typing import Dict, List, Optional
from toolpickr.core.tool import ToolDefinition

class ToolRegistry:
    def __init__(self, debug: bool = False):
        # Internal dict mapping the tool's name to its definition
        self._tools: Dict[str, ToolDefinition] = {}
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[ToolRegistry - Debug] {msg}")

    def register_tool(self, tool: ToolDefinition) -> None:
        """Adds a tool to the registry. Overwrites if a tool with the same name already exists."""
        if tool.name in self._tools:
            self._log(f"Overwriting existing tool: {tool.name}")
        else:
            self._log(f"Registering new tool: {tool.name}")
        self._tools[tool.name] = tool
        
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
            self._log(f"Removed tool: {name}")
            return True
        self._log(f"Failed to remove tool (not found): {name}")
        return False
        
    def __len__(self) -> int:
        """Allows you to run `len(registry)` to see how many tools are registered."""
        return len(self._tools)
