import streamlit as st
import os
from dotenv import load_dotenv
from satisfaction_tracker import SatisfactionTracker

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="Satisfaction Tracker", layout="centered")
st.title("📊 Customer Satisfaction Tracker")

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'results' not in st.session_state:
    st.session_state.results = []

# Automatically start tracker if API key exists
if api_key:
    if st.session_state.tracker is None:
        st.session_state.tracker = SatisfactionTracker(openrouter_api_key=api_key)
        st.success("Tracker started using API key from .env.")
else:
    st.error("API Key not found in .env file!")

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
    st.info("Tracker not initialized due to missing API key.")
