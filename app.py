"""
AI Article Humanizer (Groq) with Chat History
- Paste or type your article
- Get polished, humanized text with grammar fixes
- Refine iteratively in a chat-like interface
- Deployable anywhere (Streamlit Cloud, Hugging Face, etc.)
"""

import os
import html
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set. Please add it in your .env file.")
    st.stop()

# --------------------------
# Setup Groq LLM
# --------------------------
MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

def get_llm(model_name: str):
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name,
        temperature=0.7
    )

# --------------------------
# Humanizer Function
# --------------------------
def humanize_via_groq(llm, text: str, instructions: str = "") -> str:
    base_prompt = (
        "Your job is to act as a **professional humanizer and Grammarly-like editor**. "
        "Always fix grammar, spelling, and awkward phrasing. "
        "Make the text smooth, natural, and engaging while keeping the meaning unchanged. "
        "Do not add new facts.\n\n"
    )
    if instructions.strip():
        base_prompt += f"⚡ Extra instructions: {instructions.strip()}\n\n"

    full_prompt = base_prompt + text

    response = llm.invoke([
        ("system", "You are a skilled editor. You polish articles to sound professional and human-like."),
        ("user", full_prompt),
    ])
    return response.content if hasattr(response, "content") else str(response)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI Article Humanizer (Groq)", layout="wide")

st.title("🤖 AI Article Humanizer (Groq)")
st.markdown("Polish your articles with grammar fixes + human-like flow. Now with chat-style refinements!")

# Model choice
model_choice = st.selectbox("Choose Groq Model", MODEL_CHOICES, index=1)
llm = get_llm(model_choice)

# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# First input box (article paste)
if not st.session_state.chat_history:
    original_text = st.text_area("✍️ Paste Your Article", height=250)
    if st.button("✨ Polish Article"):
        if not original_text.strip():
            st.warning("Please paste your article before running.")
        else:
            with st.spinner("Polishing with Groq..."):
                polished = humanize_via_groq(llm, original_text)
                st.session_state.chat_history.append({"role": "user", "content": original_text})
                st.session_state.chat_history.append({"role": "assistant", "content": polished})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])

# Follow-up refinements
follow_up = st.chat_input("Refine or give new instructions...")
if follow_up:
    st.session_state.chat_history.append({"role": "user", "content": follow_up})
    with st.chat_message("user"):
        st.write(follow_up)

    with st.spinner("Refining with Groq..."):
        last_ai_texts = [m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"]
        latest_text = last_ai_texts[-1] if last_ai_texts else follow_up
        polished = humanize_via_groq(llm, latest_text, instructions=follow_up)
        st.session_state.chat_history.append({"role": "assistant", "content": polished})
        with st.chat_message("assistant"):
            st.write(polished)

# Footer
st.markdown("<hr><small>Built with ❤️ — AI Article Humanizer (Groq)</small>", unsafe_allow_html=True)
