from langchain_core.messages import HumanMessage, SystemMessage
from hands_free_coding_tools.base_llm import get_llm

llm = get_llm()

SYSTEM_PROMPT = """
You are a senior software engineer and code reviewer.
You will be given one or more code chunks from a codebase and a user objective.
Explain the code clearly: what it does, how it works, and how it relates to the objective.
Be concise but thorough.
"""


def explain_tool(objective: str, chunks: list[tuple[str, dict, float]]) -> str:
    """Explains the top retrieved code chunks relative to the user objective.

    Args:
        objective: What the user wants to understand.
        chunks: List of (doc_text, metadata, score) tuples from the retriever.

    Returns:
        A string explanation from the LLM.
    """
    if not chunks:
        return "No relevant code chunks were found to explain."

    context_parts = []
    for doc, metadata, score in chunks:
        file_path  = metadata.get("file_path", "unknown")
        start_line = metadata.get("start_line", "?")
        end_line   = metadata.get("end_line",   "?")
        context_parts.append(
            f"### {file_path} (lines {start_line}–{end_line})\n```python\n{doc}\n```"
        )

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"**Objective / Question:** {objective}\n\n"
            f"**Relevant code:**\n\n{context}"
        )),
    ]

    return llm.invoke(messages).content
