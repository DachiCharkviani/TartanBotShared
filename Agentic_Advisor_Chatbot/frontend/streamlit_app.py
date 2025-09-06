import sys, os, time, threading, queue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import streamlit as st
from uuid import uuid4

API_BASE = (os.getenv("API_BASE")).rstrip("/")

st.set_page_config(page_title="Agentic Advisor Chatbot", layout="centered")
# --- HIDE TOP-RIGHT UI (Deploy button, ⋮ menu, etc.) ---
st.markdown("""
<style>
/* Hide Streamlit's top header toolbar */
header {visibility: hidden;}
/* Also hide the legacy hamburger menu and footer just in case */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Extra safety for newer builds */
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
</style>
""", unsafe_allow_html=True)
st.title("Agentic Advisor - TartanBot")

# --- Session state setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list[{"role": "user"|"assistant", "content": str}]

# --- Render existing history first ---
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

# --- Input ---
user_input = st.chat_input("Ask me about CMUQ courses...")

if user_input:
    # 1) Show the user's message *immediately* and persist it
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2) Create an assistant placeholder while we wait for the backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("_thinking…_")

    # 3) Call backend (blocking is fine; UI already shows user msg + placeholder)
    payload = {"session_id": st.session_state.session_id, "message": user_input}

   # --- ANIMATION + CONCURRENT BACKEND CALL ---
    # Run the blocking HTTP request in a background thread and animate the placeholder on the main thread.
    q: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def fetch_reply():
        try:
            r = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"Backend error: {r.text}")

            # Try JSON first, then fall back to raw text
            try:
                data = r.json()
                reply_text = (
                    data.get("reply") if isinstance(data, dict)
                    else (data if isinstance(data, str) else str(data))
                )
            except Exception:
                reply_text = r.text or ""

            q.put(("ok", reply_text))
        except Exception as e:
            q.put(("err", f"⚠️ {type(e).__name__}: {e}"))

    t = threading.Thread(target=fetch_reply, daemon=True)
    t.start()

    dots = 1
    start = time.time()
    max_wait_sec = 125  # a hair over the request timeout
    status, result_text = None, None

    # Animate until the background thread posts a result to the queue
    while True:
        try:
            status, result_text = q.get_nowait()
            break
        except queue.Empty:
            message_placeholder.markdown(f"_thinking{'.' * dots}_")
            dots = 1 if dots == 3 else dots + 1
            time.sleep(0.5)  # controls the speed of the dot animation
            if time.time() - start > max_wait_sec:
                status, result_text = "err", "⚠️ Timeout waiting for the backend."
                break

    # 4) Update the placeholder with the actual reply, then persist to history
    if status == "ok":
        message_placeholder.write(result_text)
        st.session_state.history.append({"role": "assistant", "content": result_text})
    else:
        message_placeholder.error(result_text)
        st.session_state.history.append({"role": "assistant", "content": result_text})
