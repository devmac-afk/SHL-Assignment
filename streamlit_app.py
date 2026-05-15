"""
streamlit_app.py — Simple Streamlit chat UI for the SHL RAG Recommender.
"""

import os

import requests
import streamlit as st

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 SHL Assessment Recommender")
st.caption("Describe the role you're hiring for, and I'll recommend the best SHL assessments.")

# ── Session state ─────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": ..., "content": ..., "recommendations": ...}

# ── Sidebar controls ──────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example prompts")
    st.markdown("""
- *I need to hire a mid-level Java developer*
- *We are looking for a Sales Manager with leadership skills*
- *Recommend cognitive ability tests for data analysts*
- *What personality assessments does SHL offer?*
""")

# ── Display conversation ──────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show recommendation cards if present
        if msg.get("recommendations"):
            st.markdown("**📋 Recommended Assessments:**")
            for rec in msg["recommendations"]:
                name = rec.get("name", "")
                url = rec.get("url", "#")
                test_type = rec.get("test_type", "")
                st.markdown(f"- **[{name}]({url})** — *{test_type}*")

# ── Chat input ────────────────────────────────

user_input = st.chat_input("Ask about SHL assessments...")

if user_input:
    # Show the user's message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "recommendations": [],
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Build the payload (full conversation history)
    payload_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Finding assessments..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"messages": payload_messages},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                reply = data.get("reply", "Sorry, I could not generate a response.")
                recommendations = data.get("recommendations", [])

                st.write(reply)

                if recommendations:
                    st.markdown("**📋 Recommended Assessments:**")
                    for rec in recommendations:
                        name = rec.get("name", "")
                        url = rec.get("url", "#")
                        test_type = rec.get("test_type", "")
                        st.markdown(f"- **[{name}]({url})** — *{test_type}*")

                # Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "recommendations": recommendations,
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend. Make sure `uvicorn app:app` is running.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The server is taking too long.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
