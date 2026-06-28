from state import AgentState                          # fixed: was from AgentGraph
from hands_free_coding_tools.retriever_tool import retriever
from hands_free_coding_tools.explain_tool import explain_tool


def explain_node(state: AgentState):
    reranked_results = retriever(state["objective"])
    explanation      = explain_tool(state["objective"], reranked_results)
    return {
        "generated_code": explanation,
        "status":         "done",
    }
