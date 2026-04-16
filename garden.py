import json
import uuid
import time
from pathlib import Path
from datetime import datetime

import ollama
import streamlit as st

MODEL = "gemma4:31b-cloud"
SYSTEM_PROMPT = ""  # Add your system prompt here
MAX_CONTEXT_MESSAGES = 16
THREADS_DIR = Path("threads")
THREADS_DIR.mkdir(exist_ok=True)


def load_all_threads() -> dict:
    threads = {}
    for f in sorted(THREADS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            threads[data["id"]] = data
        except Exception:
            pass
    return threads


def save_thread(thread: dict):
    path = THREADS_DIR / f"{thread['id']}.json"
    path.write_text(json.dumps(thread, ensure_ascii=False, indent=2))


def delete_thread(thread_id: str):
    path = THREADS_DIR / f"{thread_id}.json"
    path.unlink(missing_ok=True)


def new_thread(title: str = "New Chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


def build_ollama_messages(messages: list) -> list:
    result = []
    if SYSTEM_PROMPT:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    for m in messages[-MAX_CONTEXT_MESSAGES:]:
        entry = {"role": m["role"], "content": m["content"]}
        if m.get("images"):
            entry["images"] = m["images"]
        result.append(entry)
    return result


def init_state():
    if "threads" not in st.session_state:
        st.session_state.threads = load_all_threads()
    if "active_thread_id" not in st.session_state:
        ids = list(st.session_state.threads.keys())
        st.session_state.active_thread_id = ids[0] if ids else None
    if "renaming_thread_id" not in st.session_state:
        st.session_state.renaming_thread_id = None


def get_active_thread() -> dict | None:
    tid = st.session_state.active_thread_id
    return st.session_state.threads.get(tid) if tid else None


def create_new_thread():
    t = new_thread()
    st.session_state.threads[t["id"]] = t
    st.session_state.active_thread_id = t["id"]
    save_thread(t)


def switch_thread(tid: str):
    st.session_state.active_thread_id = tid
    st.session_state.renaming_thread_id = None


def handle_rename_submit(tid: str, new_title: str):
    if new_title.strip():
        st.session_state.threads[tid]["title"] = new_title.strip()
        save_thread(st.session_state.threads[tid])
    st.session_state.renaming_thread_id = None


def handle_delete(tid: str):
    del st.session_state.threads[tid]
    delete_thread(tid)
    ids = list(st.session_state.threads.keys())
    st.session_state.active_thread_id = ids[0] if ids else None


init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Threads")
st.sidebar.button("New Chat", type="primary", use_container_width=True, on_click=create_new_thread)
st.sidebar.divider()

for tid, thread in list(st.session_state.threads.items()):
    is_active = tid == st.session_state.active_thread_id
    is_renaming = st.session_state.renaming_thread_id == tid

    if is_renaming:
        new_title = st.sidebar.text_input(
            "Rename",
            value=thread["title"],
            key=f"rename_input_{tid}",
            label_visibility="collapsed",
        )
        rcol1, rcol2 = st.sidebar.columns(2)
        if rcol1.button("Save", key=f"rename_save_{tid}", use_container_width=True):
            handle_rename_submit(tid, new_title)
            st.rerun()
        if rcol2.button("Cancel", key=f"rename_cancel_{tid}", use_container_width=True):
            st.session_state.renaming_thread_id = None
            st.rerun()
    else:
        col1, col2 = st.sidebar.columns([5, 1])
        btn_type = "secondary" if is_active else "tertiary"
        if col1.button(thread["title"], key=f"thread_btn_{tid}", type=btn_type, icon="💬", use_container_width=True):
            switch_thread(tid)
            st.rerun()
        action = col2.menu_button("", options=["Rename", "Delete"], type="tertiary", key=f"menu_{tid}")
        if action == "Rename":
            st.session_state.renaming_thread_id = tid
            st.rerun()
        elif action == "Delete":
            handle_delete(tid)
            st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────

thread = get_active_thread()

if thread is None:
    st.markdown("## No threads yet")
    st.markdown("Create a new chat using the sidebar.")
    st.stop()

# Editable title
new_title_val = st.text_input(
    "Thread title",
    value=thread["title"],
    key=f"title_input_{thread['id']}",
    label_visibility="collapsed",
)
if new_title_val != thread["title"]:
    thread["title"] = new_title_val
    st.session_state.threads[thread["id"]]["title"] = new_title_val
    save_thread(thread)

st.divider()

# Render existing messages
for msg in thread["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("files"):
            for fname in msg["files"]:
                st.caption(f"📎 {fname}")

# Chat input
prompt_input = st.chat_input(
    "Message…",
    accept_file="multiple",
    accept_audio=True,
    key=f"chat_input_{thread['id']}",
)

if prompt_input:
    text = prompt_input.text or ""
    uploaded_files = prompt_input.files or []

    file_names = []
    images_b64 = []

    for f in uploaded_files:
        file_names.append(f.name)
        mime = getattr(f, "type", "") or ""
        if mime.startswith("image/"):
            import base64
            images_b64.append(base64.b64encode(f.read()).decode())

    if not text and file_names:
        text = f"[Attached: {', '.join(file_names)}]"

    user_msg: dict = {"role": "user", "content": text, "files": file_names}
    if images_b64:
        user_msg["images"] = images_b64

    thread["messages"].append(user_msg)
    save_thread(thread)

    with st.chat_message("user"):
        st.markdown(text)
        for fname in file_names:
            st.caption(f"📎 {fname}")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        ollama_msgs = build_ollama_messages(thread["messages"][:-1] + [user_msg])

        try:
            stream = ollama.chat(
                model=MODEL,
                messages=ollama_msgs,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.message.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

    assistant_msg = {"role": "assistant", "content": full_response}
    thread["messages"].append(assistant_msg)
    save_thread(thread)
    st.rerun()
