from langchain_core.messages import HumanMessage, SystemMessage
from hands_free_coding_tools.base_llm import get_llm
from pydantic import BaseModel
from typing import Literal


class PlannerSchema(BaseModel):
    workflow: Literal["new_project", "modify_project", "explain_code", "none"]
    objective: str
    codebase_exists: bool


llm_planner = get_llm().with_structured_output(
    PlannerSchema, method="json_schema", strict=True
)

SYSTEM_PROMPT = """
You are a strict workflow router for a coding assistant. Your job is to classify
the user's query into exactly one workflow and set codebase_exists correctly.

WORKFLOW DEFINITIONS

new_project:
  The user wants code written from scratch with NO existing codebase involved.
  Use for ANY coding or programming request where the user is not referring to
  an existing project they own.

  Examples:
  - "write a bubble sort in python"
  - "create a function to reverse a linked list"
  - "write a script that reads a CSV file"
  - "build a tic tac toe game"
  - "code a binary search algorithm"
  - "write me a FastAPI hello world"

modify_project:
  The user wants to change, extend, fix, or add to an EXISTING codebase they own.
  Only use this when the user explicitly refers to their own existing code.

  Trigger phrases: "my codebase", "my project", "my file", "this project",
  "existing code", "add to my", "fix my", "update my", "in my app".

  Examples:
  - "add a /health endpoint to my FastAPI app"
  - "fix the bug in my read_tool.py"
  - "refactor my retriever node"

explain_code:
  The user wants to understand code from their existing codebase.
  Only use when they refer to code they have uploaded or own.

  Examples:
  - "explain how my retriever_node works"
  - "what does my planner_node do"

none:
  Use ONLY for queries that have absolutely nothing to do with coding or software.
  General knowledge, sports, people, weather, opinions — anything non-technical.

  Examples:
  - "who is Lionel Messi"
  - "what is the capital of France"
  - "how are you"
  - "tell me a joke"

CRITICAL RULES — NEVER BREAK THESE

1. If the query is any kind of coding or programming request, NEVER return none.
   Even vague coding requests like "write something in python" → new_project.

2. If codebase_exists is False → workflow MUST be new_project or none.
   NEVER return modify_project or explain_code when codebase_exists is False.

3. codebase_exists is True ONLY when the user explicitly references
   their own existing project, file, or codebase. A generic coding
   request does NOT imply codebase_exists = True.

4. When in doubt between new_project and modify_project → choose new_project.

OUTPUT FIELDS

workflow:        one of new_project | modify_project | explain_code | none
codebase_exists: true only if user references their own existing code
objective:       a clear concise prompt for the downstream code-generation LLM
                 capturing exactly what needs to be built or done.
                 For none workflow set objective to empty string.

Return only the schema. No explanation.
"""


def planner_tool(query: str) -> PlannerSchema:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    return llm_planner.invoke(messages)