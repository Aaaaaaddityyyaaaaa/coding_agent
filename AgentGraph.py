from langgraph.graph import StateGraph, START, END
from state import AgentState                          # no longer defines AgentState here
from new_project_nodes import generate_new_node, save_new_node
from planner_node import planner_node
from modify_code_base_nodes import (
    modify_router_node, read_node, embed_node, retriever_node, modify_file_node
)
from explain_node import explain_node


# ── Routing functions (pure functions — NOT graph nodes) ──────────────────────

def router_node(state: AgentState) -> str:
    return state["workflow"]

def reviewer_node(state: AgentState) -> str:
    if state["trys"] > 3:
        return "done"
    return state["status"]

def modify_router(state: AgentState) -> str:
    return state["workflow_modify"]


# ── Build graph ───────────────────────────────────────────────────────────────

Graph = StateGraph(state_schema=AgentState)

Graph.add_node("planner_node",      planner_node)
Graph.add_node("generate_new_node", generate_new_node)
Graph.add_node("save_new_node",     save_new_node)
Graph.add_node("reader_node",       read_node)
Graph.add_node("embed_node",        embed_node)
Graph.add_node("modify_router",     modify_router_node)
Graph.add_node("retriever_node",    retriever_node)
Graph.add_node("modify_file_node",  modify_file_node)
Graph.add_node("explain_node",      explain_node)


# ── Edges ─────────────────────────────────────────────────────────────────────

Graph.add_edge(START, "planner_node")

Graph.add_conditional_edges("planner_node", router_node, {
    "new_project":    "generate_new_node",
    "modify_project": "reader_node",
    "explain_code":   "explain_node",
    "none":           END,
})

# new_project flow
Graph.add_edge("generate_new_node", "save_new_node")
Graph.add_conditional_edges("save_new_node", reviewer_node, {
    "done":   END,
    "failed": "planner_node",
})

# modify_project flow
Graph.add_edge("reader_node",  "embed_node")
Graph.add_edge("embed_node",   "modify_router")
Graph.add_conditional_edges("modify_router", modify_router, {
    "new_file":    "generate_new_node",
    "modify_file": "retriever_node",
    "none":        END,
})
Graph.add_edge("retriever_node",   "modify_file_node")
Graph.add_edge("modify_file_node", END)

# explain_code flow
Graph.add_edge("explain_node", END)


# ── Compile ───────────────────────────────────────────────────────────────────

graph = Graph.compile()
