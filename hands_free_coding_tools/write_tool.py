from pathlib import Path
from langchain_core.tools import tool


@tool
def write_new(path: str, code: str):
    """Creates a new file and writes code into it. Fails if file already exists."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        return f"File exists"
    file.write_text(code, encoding="utf-8")
    return f"Created {path}"


@tool
def patch(path: str, start_line: int, end_line: int, new_chunk: str):
    """Replaces lines start_line to end_line (exclusive) in a file with new_chunk."""
    file = Path(path)
    if not file.exists():
        return "File Doesn't Exist"
    lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[start_line:end_line] = [new_chunk]
    file.write_text("".join(lines), encoding="utf-8")
    return f"Patched {path} lines {start_line}-{end_line}"


@tool
def append(path: str, code: str):
    """Appends code to the end of an existing file."""
    file = Path(path)
    if not file.exists():
        return "File Doesn't Exist"
    with open(path, "a", encoding="utf-8") as f:
        f.write(code)
    return f"Appended to {path}"
