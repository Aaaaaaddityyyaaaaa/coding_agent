from typing import TypedDict, Literal, Annotated
import operator


class AgentState(TypedDict):
    query:            str
    workflow:         Literal["new_project", "modify_project", "explain_code", "none"]
    objective:        str
    codebase_exists:  bool
    workflow_modify:  Literal["new_file", "modify_file", "none"]
    root_path:        str
    python_chunks:    list
    java_chunks:      list
    js_chunks:        list
    c_chunks:         list
    new_file_name:    str
    retrieved_chunks: list
    generated_code:   str
    target_file:      str
    start_line:       int
    end_line:         int
    old_chunk:        str
    trys:             int
    messages:         Annotated[list, operator.add]
    error:            str
    status:           Literal["planning", "reading", "retrieving",
                              "generating", "writing", "done", "failed"]
