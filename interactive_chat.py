import streamlit as st
import os
from satisfaction_tracker import SatisfactionTracker  # تأكد أن الملف موجود في نفس المجلد

st.set_page_config(page_title="Satisfaction Tracker", layout="centered")
st.title("📊 Customer Satisfaction Tracker")

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'results' not in st.session_state:
    st.session_state.results = []

# Input API Key
api_key = st.text_input("🔑 Enter your OpenRouter API Key", type="password")
if st.button("Start Tracker"):
    if api_key.strip() == "":
        st.warning("Please enter a valid API key.")
    else:
        st.session_state.tracker = SatisfactionTracker(openrouter_api_key=api_key.strip())
        st.session_state.chat_history.clear()
        st.session_state.results.clear()
        st.success("Tracker started. You can now send messages.")

if st.session_state.tracker:
    st.subheader("💬 Chat Interface")
    col1, col2 = st.columns([1, 4])
    role = col1.selectbox("Role", ["user", "assistant"])
    message = col2.text_input("Message", key="msg_input")

    if st.button("Send Message"):
        if message.strip() != "":
            result = st.session_state.tracker.add_message(role, message.strip())
            st.session_state.chat_history.append((role, message.strip()))
            st.session_state.results.append(result)
        else:
            st.warning("Message cannot be empty.")

    # Show chat history
    st.markdown("---")
    st.subheader("🗨️ Conversation")
    for role, msg in st.session_state.chat_history:
        align = "👤" if role == "user" else "🤖"
        with st.chat_message(role):
            st.markdown(f"{align} **{role.capitalize()}**: {msg}")

    # Show results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📈 Satisfaction Updates")
        for res in reversed(st.session_state.results):
            st.success(f"Updated Score: {res['updated_score']}/5 | Status: {res['status']}")
            st.caption(f"Reason: {res['reason']}")
else:
    st.info("Enter API key and click Start to begin.")
