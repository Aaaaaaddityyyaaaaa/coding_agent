from langchain_core.messages import HumanMessage, SystemMessage
from hands_free_coding_tools.base_llm import get_llm
from pydantic import BaseModel
from typing import Literal


class ModifyRouterSchema(BaseModel):
    workflow: Literal["new_file", "modify_file", "none"]
    objective: str
    new_file_name: str   # empty string "" when not creating a new file


llm_router = get_llm().with_structured_output(
    ModifyRouterSchema, method="json_schema", strict=True
)

SYSTEM_PROMPT = """
You are a workflow router for codebase modifications.

new_file:
  Use when the user wants to create an entirely NEW file inside an existing project.
  Examples: "Add a new routes file", "Create a utils module"

modify_file:
  Use when the user wants to CHANGE an existing file.
  Includes: editing, adding functions, fixing bugs, refactoring, appending.
  If user mentions "my file", a specific filename, "existing code", or passes an old chunk → ALWAYS choose modify_file.

none:
  Use if neither applies.

new_file_name:
  - Provide a suitable filename WITH extension when workflow is "new_file"  (e.g. "calculator.py")
  - Return empty string "" for any other workflow

objective: concise prompt for the downstream code-generation LLM.

Return only the schema.
"""


def modify_router_tool(query: str) -> ModifyRouterSchema:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    return llm_router.invoke(messages)
