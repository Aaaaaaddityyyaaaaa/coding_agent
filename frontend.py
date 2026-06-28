"""
Streamlit frontend for the interview_llm agentic coding assistant.

Run alongside the FastAPI backend:
  uvicorn backend:app --reload          # terminal 1
  streamlit run frontend.py             # terminal 2
"""

import streamlit as st
import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="interview_llm",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 interview_llm")
st.caption("Hands-free agentic coding assistant — upload your project and describe what you want.")

# ── Session state ─────────────────────────────────────────────────────────────

if "session_id"  not in st.session_state:
    st.session_state.session_id  = None
if "root_path"   not in st.session_state:
    st.session_state.root_path   = None
if "history"     not in st.session_state:
    st.session_state.history     = []     # list of {query, result} dicts


# ── Sidebar — project upload ───────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Project")

    uploaded_zip = st.file_uploader(
        "Upload your codebase (.zip)",
        type=["zip"],
        help="Zip your project folder and upload it here.",
    )

    if uploaded_zip and st.button("Upload project", type="primary"):
        with st.spinner("Uploading and extracting…"):
            response = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_zip.name, uploaded_zip.getvalue(), "application/zip")},
            )
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.root_path  = data["root_path"]
            st.success("Project uploaded ✅")
        else:
            st.error(f"Upload failed: {response.text}")

    # Show current session info
    if st.session_state.session_id:
        st.divider()
        st.markdown("**Active session**")
        st.code(st.session_state.session_id[:8] + "…", language=None)
        st.markdown("**Root path**")
        st.code(st.session_state.root_path, language=None)

        if st.button("🗑️ Clear session", type="secondary"):
            requests.delete(f"{API_BASE}/sessions/{st.session_state.session_id}")
            st.session_state.session_id = None
            st.session_state.root_path  = None
            st.session_state.history    = []
            st.rerun()
    else:
        st.info("No project loaded yet.")

    st.divider()
    st.markdown("**Backend**")
    try:
        health = requests.get(f"{API_BASE}/sessions", timeout=2)
        st.success("Online ✅")
    except Exception:
        st.error("Offline ❌ — start `uvicorn backend:app --reload`")


# ── Main area ─────────────────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 Request")

    query = st.text_area(
        "Describe what you want to do",
        height=160,
        placeholder=(
            "Examples:\n"
            "• Create a Tic Tac Toe game in Python\n"
            "• Add a /health endpoint to my FastAPI app\n"
            "• Explain how the retriever_node works\n"
            "• Fix the bug in my read_tool.py"
        ),
    )

    run_btn = st.button(
        "▶ Run agent",
        type="primary",
        disabled=(not query.strip()),
        use_container_width=True,
    )

    if run_btn:
        if not query.strip():
            st.warning("Please enter a request.")
        else:
            with st.spinner("Agent is thinking…"):
                payload = {
                    "query":      query,
                    "session_id": st.session_state.session_id,
                }
                try:
                    response = requests.post(
                        f"{API_BASE}/run",
                        json=payload,
                        timeout=300,      # agent can take a while
                    )
                    result = response.json()
                except requests.exceptions.Timeout:
                    st.error("Request timed out — agent took too long.")
                    result = None
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    result = None

            if result:
                st.session_state.history.append({"query": query, "result": result})
                st.rerun()

with col_output:
    st.subheader("📤 Output")

    if not st.session_state.history:
        st.info("Results will appear here after you run the agent.")
    else:
        # Show the most recent result at the top
        latest = st.session_state.history[-1]
        result = latest["result"]

        status   = result.get("status",   "unknown")
        workflow = result.get("workflow",  "—")
        output   = result.get("generated_code", "")
        target   = result.get("target_file", "")
        trys     = result.get("trys", 0)
        error    = result.get("error", "")

        # Status badge
        badge_color = {"done": "🟢", "failed": "🔴", "unknown": "🟡"}.get(status, "🟡")
        st.markdown(f"{badge_color} **Status:** `{status}`  |  **Workflow:** `{workflow}`  |  **Attempts:** `{trys}`")

        if target:
            st.markdown(f"**Target file:** `{target}`")

        if error:
            st.error(f"Error: {error}")

        if output:
            # Detect whether output is code or explanation text
            is_code = workflow in ("new_project", "modify_project")
            if is_code:
                st.code(output, language="python")
            else:
                st.markdown(output)

            # Download button for generated code
            if is_code and target:
                st.download_button(
                    label="⬇️ Download generated file",
                    data=output,
                    file_name=Path(target).name,
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.warning("Agent returned no output.")


# ── History ───────────────────────────────────────────────────────────────────

if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("🕘 History")

    for i, entry in enumerate(reversed(st.session_state.history[:-1]), start=1):
        with st.expander(f"[{i}] {entry['query'][:80]}…", expanded=False):
            res = entry["result"]
            st.markdown(f"**Status:** `{res.get('status')}` | **Workflow:** `{res.get('workflow')}`")
            out = res.get("generated_code", "")
            if out:
                if res.get("workflow") in ("new_project", "modify_project"):
                    st.code(out, language="python")
                else:
                    st.markdown(out)
