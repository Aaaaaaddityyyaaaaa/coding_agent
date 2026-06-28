from state import AgentState                          # fixed: was from AgentGraph
from hands_free_coding_tools.planner import planner_tool


def planner_node(state: AgentState):
    result = planner_tool(state["query"])
    return {
        "workflow":        result.workflow,
        "objective":       result.objective,
        "codebase_exists": result.codebase_exists,
        "trys":            state["trys"] + 1 if "trys" in state else 1,
    }
