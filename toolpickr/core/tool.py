# Currently Google formatted

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Property(BaseModel):
    type: str = Field(..., description="The data type of the property (e.g., 'string', 'array', 'integer')")
    description: Optional[str] = Field(default=None, description="Description of the property")
    items: Optional[Dict[str, Any]] = Field(default=None, description="Schema for the items if type is 'array'")
    
class Parameters(BaseModel):
    type: str = Field(default="object", description="Type of the parameter, usually 'object'")
    properties: Dict[str, Property] = Field(default_factory=dict, description="Dictionary defining the properties of the object")
    required: Optional[List[str]] = Field(default_factory=list, description="List of required property names")

class ToolDefinition(BaseModel):
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: Optional[Parameters] = Field(default=None, description="A schema representing the JSON schema of the function inputs")
    returns: Optional[str] = Field(default=None, description="The return value of the function")
    category: Optional[str] = Field(default=None, description="The category of the tool")
