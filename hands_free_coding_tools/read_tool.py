import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_c as tsc
import tree_sitter_javascript as tsjs
import tree_sitter_java as tsjava
from pathlib import Path

EXTENSION_LIST = [".py", ".js", ".java", ".c"]
IGNORE_LIST    = ["venv", ".venv", ".env", ".git", "__pycache__",
                  "etc", "Include", "Lib", "Scripts", "share"]

PY_LANG   = Language(tspython.language())
JAVA_LANG = Language(tsjava.language())
JS_LANG   = Language(tsjs.language())
C_LANG    = Language(tsc.language())

python_parser = Parser(PY_LANG)
java_parser   = Parser(JAVA_LANG)
js_parser     = Parser(JS_LANG)
c_parser      = Parser(C_LANG)


def extract_chunks(node, source_bytes, file_path, target_types):
    chunks = []
    if node.type in target_types:
        code      = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        name_node = node.child_by_field_name("name")
        name      = name_node.text.decode("utf-8") if name_node else "unknown"
        chunks.append({
            "code":       code,
            "name":       name,
            "type":       node.type,
            "file_path":  file_path,
            "start_line": node.start_point[0],
            "end_line":   node.end_point[0],
        })
    for child in node.children:
        chunks.extend(extract_chunks(child, source_bytes, file_path, target_types))
    return chunks


def create_chunks_python(path):
    source_bytes = Path(path).read_bytes()
    tree = python_parser.parse(source_bytes)
    return extract_chunks(tree.root_node, source_bytes, path,
                          {"function_definition", "class_definition"})


def create_chunks_js(path):
    source_bytes = Path(path).read_bytes()
    tree = js_parser.parse(source_bytes)
    return extract_chunks(tree.root_node, source_bytes, path,
                          {"function_declaration", "arrow_function", "class_declaration"})


def create_chunks_java(path):
    source_bytes = Path(path).read_bytes()
    tree = java_parser.parse(source_bytes)
    return extract_chunks(tree.root_node, source_bytes, path,
                          {"method_declaration", "class_declaration"})


def create_chunks_c(path):
    source_bytes = Path(path).read_bytes()
    tree = c_parser.parse(source_bytes)
    return extract_chunks(tree.root_node, source_bytes, path,
                          {"function_definition"})


def reader(root: str):
    """Walk root directory and return chunks grouped by language.
    Returns: (python_chunks, java_chunks, js_chunks, c_chunks)
    """
    python_chunks, java_chunks, js_chunks, c_chunks = [], [], [], []
    all_paths = []

    for current_path, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in EXTENSION_LIST:
                all_paths.append(os.path.join(current_path, file))

    for path in all_paths:
        ext = os.path.splitext(path)[1]
        if ext == ".py":
            python_chunks.extend(create_chunks_python(path))
        elif ext == ".js":
            js_chunks.extend(create_chunks_js(path))
        elif ext == ".java":
            java_chunks.extend(create_chunks_java(path))
        elif ext == ".c":
            c_chunks.extend(create_chunks_c(path))

    # Order matches what read_node expects: python, java, js, c
    return python_chunks, java_chunks, js_chunks, c_chunks