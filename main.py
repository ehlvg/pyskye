import base64
import json
import subprocess
import uuid
from pathlib import Path
from datetime import datetime

import ollama
import streamlit as st

MODEL = "gemma4:31b-cloud"
SYSTEM_PROMPT = ""  # Add your system prompt here
MAX_CONTEXT_MESSAGES = 16
THREADS_DIR = Path("threads")
THREADS_DIR.mkdir(exist_ok=True)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute one or more bash commands on the user's machine. "
            "Use this to read files, run scripts, check system state, install packages, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of bash commands to run sequentially. "
                        "Each element must be a complete shell command string, e.g. "
                        '[\"mkdir -p ~/Notes\", \"echo hello > ~/Notes/hi.md\"]'
                    ),
                }
            },
            "required": ["commands"],
        },
    },
}


def run_bash(commands: list[str]) -> str:
    outputs = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = result.stdout
            err = result.stderr
            combined = out
            if err:
                combined += ("\n" if combined else "") + err
            outputs.append(f"$ {cmd}\n{combined.rstrip()}")
        except subprocess.TimeoutExpired:
            outputs.append(f"$ {cmd}\n[timeout after 60s]")
        except Exception as e:
            outputs.append(f"$ {cmd}\n[error: {e}]")
    return "\n\n".join(outputs)


# ── Persistence ───────────────────────────────────────────────────────────────

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
    (THREADS_DIR / f"{thread_id}.json").unlink(missing_ok=True)


def new_thread(title: str = "New Chat") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }


# ── Ollama helpers ────────────────────────────────────────────────────────────

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


def render_stored_message(msg: dict):
    """Render a message that was already stored (history display)."""
    role = msg["role"]

    if msg.get("type") == "tool_call":
        with st.expander(f"🔧 bash — {len(msg.get('commands', []))} command(s)", expanded=False):
            for cmd in msg.get("commands", []):
                st.code(cmd, language="bash")
        return

    if msg.get("type") == "tool_result":
        with st.expander("📤 Output", expanded=False):
            st.code(msg["content"], language="text")
        return

    with st.chat_message(role):
        if msg["content"]:
            st.markdown(msg["content"])
        if msg.get("files"):
            for fname in msg["files"]:
                st.caption(f"📎 {fname}")


def run_agentic_loop(ollama_msgs: list, thread: dict):
    """
    Drive the model → tool → model loop.
    Yields nothing; writes directly to the Streamlit UI and appends to thread["messages"].
    """
    while True:
        # Non-streaming call so we can inspect tool_calls
        response = ollama.chat(
            model=MODEL,
            messages=ollama_msgs,
            tools=[BASH_TOOL],
            stream=False,
        )
        msg = response.message
        text_content = msg.content or ""
        tool_calls = msg.tool_calls or []

        if text_content:
            with st.chat_message("assistant"):
                st.markdown(text_content)
            thread["messages"].append({"role": "assistant", "content": text_content})
            ollama_msgs.append({"role": "assistant", "content": text_content})

        if not tool_calls:
            break

        for tc in tool_calls:
            fn = tc.function
            args = fn.arguments if isinstance(fn.arguments, dict) else json.loads(fn.arguments)
            raw_commands = args.get("commands", args.get("command", []))
            if isinstance(raw_commands, str):
                commands = [raw_commands]
            elif isinstance(raw_commands, list):
                commands = [str(c) for c in raw_commands if str(c).strip()]
            else:
                commands = [str(raw_commands)]

            # Show tool call
            with st.expander(f"🔧 bash — {len(commands)} command(s)", expanded=True):
                for cmd in commands:
                    st.code(cmd, language="bash")

            thread["messages"].append({
                "role": "tool_call",
                "type": "tool_call",
                "content": "",
                "commands": commands,
            })

            # Execute
            output = run_bash(commands)

            # Show output
            with st.expander("📤 Output", expanded=True):
                st.code(output, language="text")

            thread["messages"].append({
                "role": "tool_result",
                "type": "tool_result",
                "content": output,
            })

            # Feed result back as tool message
            ollama_msgs.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            ollama_msgs.append({"role": "tool", "content": output})

        save_thread(thread)


# ── Session state ─────────────────────────────────────────────────────────────

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

# ── Sidebar ───────────────────────────────────────────────────────────────────

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

# ── Main area ─────────────────────────────────────────────────────────────────

thread = get_active_thread()

if thread is None:
    st.markdown("## No threads yet")
    st.markdown("Create a new chat using the sidebar.")
    st.stop()

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

for msg in thread["messages"]:
    render_stored_message(msg)

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

    ollama_msgs = build_ollama_messages(thread["messages"])

    try:
        run_agentic_loop(ollama_msgs, thread)
    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown(f"⚠️ Error: {e}")
        thread["messages"].append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        save_thread(thread)

    st.rerun()
