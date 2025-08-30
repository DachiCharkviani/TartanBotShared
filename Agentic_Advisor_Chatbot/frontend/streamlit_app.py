import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import streamlit as st
from uuid import uuid4
# from backend.settings import API_HOST, API_PORT

API_BASE = os.getenv("API_BASE").rstrip("/")
st.set_page_config(page_title="Agentic Advisor Chatbot", layout="centered")

st.title("Agentic Advisor Chatbot")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())





chat_placeholder = st.container()
user_input = st.chat_input("Ask me about CMUQ courses...")
#st.markdown("Chat with powerful LLMs like Groq and OpenAI GPT-4")

if user_input:
    payload = {"session_id": st.session_state.session_id, "message": user_input}
    r = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    if r.status_code != 200:
        st.error(f"Backend error: {r.text}")
    else:
        
        reply_text = ""
        try:
            data = r.json()
            if isinstance(data, dict) and "reply" in data:
                reply_text = data["reply"]
            elif isinstance(data, str):
                reply_text = data
            else:
                reply_text = str(data)
        except Exception:
            reply_text = r.text or ""
        
        st.session_state.setdefault("history", []).append({"role": "user", "content": user_input})
        st.session_state["history"].append({"role": "assistant", "content": reply_text})


with chat_placeholder:
    for turn in st.session_state.get("history", []):
        with st.chat_message(turn["role"]):
            st.write(turn["content"])


st.caption("Powered by LangGraph + Chroma + FastAPI. Switch to any frontend by calling the same /chat API.")

