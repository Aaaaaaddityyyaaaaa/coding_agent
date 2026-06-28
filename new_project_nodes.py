import os
from state import AgentState                          # fixed: was from AgentGraph
from hands_free_coding_tools.generate_tool import generate_tool
from hands_free_coding_tools.write_tool import write_new


def generate_new_node(state: AgentState):
    response    = generate_tool.invoke({"prompt": state["objective"]})
    name        = state.get("new_file_name", "") or "new.py"
    root        = state.get("root_path", "")

    # fixed: prefix with root_path so file lands inside the uploaded project
    target_file = os.path.join(root, name) if root else name

    return {
        "generated_code": response,
        "target_file":    target_file,
    }


def save_new_node(state: AgentState):
    result = write_new.invoke({
        "path": state["target_file"],
        "code": state["generated_code"],
    })
    status = "done" if result.startswith("Created") else "failed"
    return {"status": status}
