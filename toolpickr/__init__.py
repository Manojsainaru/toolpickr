"""
ToolPickr — Smart and scalable tool picking for LLMs.

Quick start:
    from toolpickr import ToolPickr, ToolDefinition

    pickr = ToolPickr()  # Zero-config, uses local sentence-transformers
    pickr.register_tools(my_tools)
    pickr.build()

    tool_schema = pickr.get_tool(format="gemini")
    result = pickr.handle_tool_call({"action": "search", "queries": ["send email"]})
"""

from toolpickr._version import __version__
from toolpickr.pickr import ToolPickr
from toolpickr.core.tool import ToolDefinition, Parameters, Property
from toolpickr.core.results import ToolCallResult
from toolpickr.core.registry import ToolRegistry

__all__ = [
    "__version__",
    "ToolPickr",
    "ToolDefinition",
    "Parameters",
    "Property",
    "ToolCallResult",
    "ToolRegistry",
]
