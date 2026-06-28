"""
FastAPI backend for interview_llm.

Endpoints:
  POST /upload               — upload zip of codebase, returns session_id
  GET  /root_path            — returns root path as plain text (kept for compatibility)
  POST /run                  — run the agent, returns final state JSON
  GET  /sessions             — list active sessions
  DELETE /sessions/{id}      — delete a session and its files
"""

import os, zipfile, shutil, tempfile, uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from AgentGraph import graph
from state import AgentState

app = FastAPI(title="interview_llm API", version="0.1.0")

UPLOAD_DIR = Path(os.path.expanduser("~/interview_llm_uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

_sessions:     dict[str, str] = {}
_current_root: str            = ""


# ── /upload ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_project(file: UploadFile = File(...)):
    global _current_root

    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip files are accepted.")

    session_id  = str(uuid.uuid4())
    extract_dir = UPLOAD_DIR / session_id

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(tmp_path, "r") as z:
            z.extractall(extract_dir)
    finally:
        os.unlink(tmp_path)

    root_path             = str(extract_dir)
    _sessions[session_id] = root_path
    _current_root         = root_path

    return {"session_id": session_id, "root_path": root_path}


# ── /root_path ────────────────────────────────────────────────────────────────

@app.get("/root_path", response_class=PlainTextResponse)   # fixed: plain text, no JSON quotes
def get_root_path(session_id: str | None = None):
    global _current_root
    if session_id:
        if session_id not in _sessions:
            raise HTTPException(404, "Session not found.")
        return _sessions[session_id]
    if not _current_root:
        raise HTTPException(404, "No project uploaded yet.")
    return _current_root


# ── /run ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    query:      str
    session_id: str | None = None


@app.post("/run")
async def run_agent(req: RunRequest):
    global _current_root

    if req.session_id and req.session_id in _sessions:
        _current_root = _sessions[req.session_id]

    if not _current_root:
        raise HTTPException(400, "No project uploaded. POST to /upload first.")

    initial_state: AgentState = {
        "query":            req.query,
        "workflow":         "none",
        "objective":        "",
        "codebase_exists":  False,
        "workflow_modify":  "none",
        "root_path":        _current_root,   # passed directly into state
        "python_chunks":    [],
        "java_chunks":      [],
        "js_chunks":        [],
        "c_chunks":         [],
        "new_file_name":    "",
        "retrieved_chunks": [],
        "generated_code":   "",
        "target_file":      "",
        "start_line":       0,
        "end_line":         0,
        "old_chunk":        "",
        "trys":             0,
        "messages":         [],
        "error":            "",
        "status":           "planning",
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(500, str(e))

    return JSONResponse({
        "status":         final_state.get("status",         "unknown"),
        "workflow":       final_state.get("workflow",        ""),
        "objective":      final_state.get("objective",       ""),
        "generated_code": final_state.get("generated_code",  ""),
        "target_file":    final_state.get("target_file",     ""),
        "trys":           final_state.get("trys",            0),
        "error":          final_state.get("error",           ""),
    })


# ── /sessions ─────────────────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    return _sessions

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    path = _sessions.pop(session_id)
    shutil.rmtree(path, ignore_errors=True)
    return {"deleted": session_id}
