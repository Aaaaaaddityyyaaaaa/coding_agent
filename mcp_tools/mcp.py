"""
MCP server for the interview_llm coding agent tools.
Exposes all tools so Claude Desktop can orchestrate the full pipeline directly.

Run with:
    python mcp_server.py

Or via mcp CLI:
    mcp install mcp_server.py --name "coding-agent"

claude_desktop_config.json entry:
{
  "mcpServers": {
    "coding-agent": {
      "command": "D:\\interview_llm\\.interview_llm\\Scripts\\python.exe",
      "args": ["D:\\interview_llm\\mcp_server.py"]
    }
  }
}
"""

from mcp.server.fastmcp import FastMCP

# ── Import all tools ───────────────────────────────────────────────────────────
from hands_free_coding_tools.write_tool     import write_new, patch, append
from hands_free_coding_tools.generate_tool  import generate_tool
from hands_free_coding_tools.read_tool      import reader
from hands_free_coding_tools.embedding_tool import embed_chunks
from hands_free_coding_tools.retriever_tool import retriever
from hands_free_coding_tools.explain_tool   import explain_tool

mcp = FastMCP("coding-agent")


# ── File tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def create_file(path: str, code: str) -> str:
    """Create a new file and write code into it.
    Fails safely if the file already exists.

    Args:
        path: Absolute or relative path to the new file (e.g. 'src/utils.py').
        code: The full code content to write.
    """
    return write_new.invoke({"path": path, "code": code})


@mcp.tool()
def patch_file(path: str, start_line: int, end_line: int, new_chunk: str) -> str:
    """Replace a range of lines in an existing file with new code.
    Lines are zero-indexed. Replaces lines[start_line:end_line].

    Args:
        path:       Path to the file to edit.
        start_line: First line to replace (inclusive, 0-indexed).
        end_line:   Last line to replace (exclusive, 0-indexed).
        new_chunk:  Replacement code string (can span multiple lines).
    """
    return patch.invoke({"path": path, "start_line": start_line,
                         "end_line": end_line, "new_chunk": new_chunk})


@mcp.tool()
def append_to_file(path: str, code: str) -> str:
    """Append code to the end of an existing file.

    Args:
        path: Path to the file.
        code: Code string to append.
    """
    return append.invoke({"path": path, "code": code})


# ── Generation tool ────────────────────────────────────────────────────────────

@mcp.tool()
def generate_code(prompt: str) -> str:
    """Generate Python code from a natural language prompt using an LLM.
    Returns only the code block, no explanation.

    Args:
        prompt: A clear description of what code to generate.
    """
    return generate_tool.invoke({"prompt": prompt})


# ── RAG pipeline tools ─────────────────────────────────────────────────────────

@mcp.tool()
def read_codebase(root_path: str) -> dict:
    """Walk a codebase directory and extract AST-based code chunks.
    Supports Python, JavaScript, Java, and C files.
    Returns chunk counts per language and the chunks themselves.

    Args:
        root_path: Absolute path to the root of the codebase to read.
    """
    python_chunks, java_chunks, js_chunks, c_chunks = reader(root_path)
    return {
        "python_count": len(python_chunks),
        "java_count":   len(java_chunks),
        "js_count":     len(js_chunks),
        "c_count":      len(c_chunks),
        "python_chunks": python_chunks,
        "java_chunks":   java_chunks,
        "js_chunks":     js_chunks,
        "c_chunks":      c_chunks,
    }


@mcp.tool()
def embed_codebase(root_path: str) -> str:
    """Read a codebase and embed all code chunks into ChromaDB.
    Combines read + embed in one step — use this before retrieve_chunks.

    Args:
        root_path: Absolute path to the root of the codebase.
    """
    python_chunks, java_chunks, js_chunks, c_chunks = reader(root_path)
    total = len(python_chunks) + len(java_chunks) + len(js_chunks) + len(c_chunks)
    embed_chunks(python_chunks, java_chunks, js_chunks, c_chunks)
    return f"Embedded {total} chunks from {root_path} into ChromaDB."


@mcp.tool()
def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most semantically relevant code chunks from ChromaDB
    for a given query. Uses embedding similarity + CrossEncoder reranking.

    Args:
        query: Natural language description of what you're looking for.
        top_k: Number of top results to return (default 5, max 10).
    """
    top_k = min(top_k, 10)
    results = retriever(query)[:top_k]
    return [
        {
            "rank":       i + 1,
            "file_path":  metadata.get("file_path", "unknown"),
            "start_line": metadata.get("start_line", 0),
            "end_line":   metadata.get("end_line",   0),
            "score":      round(float(score), 4),
            "code":       doc,
        }
        for i, (doc, metadata, score) in enumerate(results)
    ]


@mcp.tool()
def explain_code_chunks(objective: str, query: str) -> str:
    """Retrieve relevant code chunks from the codebase and explain them
    in the context of the given objective.

    Args:
        objective: What the user wants to understand or achieve.
        query:     Search query to find the relevant code chunks.
    """
    chunks = retriever(query)
    return explain_tool(objective, chunks)


# ── Compound tools (full pipeline in one call) ─────────────────────────────────

@mcp.tool()
def create_and_save_code(prompt: str, file_path: str) -> str:
    """Generate code from a prompt and immediately save it to a new file.
    Combines generate_code + create_file in one step.

    Args:
        prompt:    Natural language description of what code to generate.
        file_path: Path where the generated file should be saved.
    """
    code   = generate_tool.invoke({"prompt": prompt})
    result = write_new.invoke({"path": file_path, "code": code})
    return f"{result}\n\nGenerated code:\n```python\n{code}\n```"


@mcp.tool()
def retrieve_and_patch(objective: str, file_path: str) -> str:
    """Find the most relevant code chunk in the codebase, generate
    updated code for it, and patch the file in place.
    Full RAG → generate → patch pipeline in one call.

    Args:
        objective: What change needs to be made (plain English).
        file_path: Path to the file to patch (used as fallback if retriever finds it).
    """
    # Retrieve most relevant chunk
    results = retriever(objective)
    if not results:
        return "No relevant chunks found in the codebase."

    doc, metadata, score = results[0]
    target    = metadata.get("file_path", file_path)
    start     = metadata.get("start_line", 0)
    end       = metadata.get("end_line",   0)

    # Generate replacement
    prompt    = (f"Objective: {objective}\n\n"
                 f"Existing code to replace:\n```python\n{doc}\n```\n\n"
                 f"Write the improved replacement code only.")
    new_code  = generate_tool.invoke({"prompt": prompt})

    # Patch the file
    result    = patch.invoke({"path": target, "start_line": start,
                              "end_line": end, "new_chunk": new_code})
    return f"{result}\n\nReplaced lines {start}–{end} in {target}."


