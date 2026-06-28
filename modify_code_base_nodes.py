from state import AgentState                          # fixed: was from AgentGraph
from hands_free_coding_tools.embedding_tool import embed_chunks
from hands_free_coding_tools.retriever_tool import retriever
from hands_free_coding_tools.read_tool import reader
from hands_free_coding_tools.write_tool import patch
from hands_free_coding_tools.modify_router import modify_router_tool


def read_node(state: AgentState):
    # fixed: use state["root_path"] directly — already set by backend initial_state
    # no need to call the FastAPI endpoint; avoids the JSON-string quoting issue too
    path = state["root_path"]

    python_chunks, java_chunks, js_chunks, c_chunks = reader(path)

    return {
        "root_path":     path,
        "python_chunks": python_chunks,
        "java_chunks":   java_chunks,
        "js_chunks":     js_chunks,
        "c_chunks":      c_chunks,
    }


def embed_node(state: AgentState):
    embed_chunks(
        state["python_chunks"],
        state["java_chunks"],
        state["js_chunks"],
        state["c_chunks"],
    )
    return {}


def retriever_node(state: AgentState):
    reranked_results = retriever(state["objective"])
    doc, metadata, score = reranked_results[0]
    return {
        "target_file": metadata["file_path"],
        "start_line":  metadata["start_line"],
        "end_line":    metadata["end_line"],
        "old_chunk":   doc,
    }


def modify_router_node(state: AgentState):
    response = modify_router_tool(state["objective"])
    if response.workflow == "new_file":
        return {
            "workflow_modify": response.workflow,
            "new_file_name":   response.new_file_name,
        }
    return {"workflow_modify": response.workflow}


def modify_file_node(state: AgentState):
    result = patch.invoke({
        "path":       state["target_file"],
        "start_line": state["start_line"],
        "end_line":   state["end_line"],
        "new_chunk":  state["generated_code"],
    })
    if result == "File Doesn't Exist":
        return {"status": "failed"}
    return {"status": "done"}
