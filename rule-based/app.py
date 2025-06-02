# streamlit_app.py
# ----------------
# A Streamlit app to deploy the Level-1 TF-IDF chatbot using a form
# so that the input field clears automatically on submission.
#
# Usage:
#   1. Ensure `chatbot_logic.py` and `streamlit_app.py` are in the same folder.
#   2. Place `clean_conversation_dataset.csv` in the `../data/` directory relative to this script.
#   3. Install dependencies:
#        pip install streamlit pandas scikit-learn
#   4. Run:
#        streamlit run streamlit_app.py

import streamlit as st
from chatbot_logic import Chatbot

# ──────────────────────────────────────────────────────────────────────────────
# 1) Streamlit Page Configuration (must be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TF-IDF (Rule Based) Chatbot",
    page_icon="💬",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Initialize Chatbot (cached for performance)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_bot():
    # Adjust csv_path if needed;
    return Chatbot(csv_path="D:/PROJECTS/Version Control/CHATBOT/building-smart-bots-from-scratch/data/clean_conversation_dataset.csv")

bot = load_bot()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Streamlit Page Content
# ──────────────────────────────────────────────────────────────────────────────

st.title("💬 Level-1 TF-IDF (Rule Based) Chatbot")
st.write(
    "Type a message below and press 'Send'. "
    "The bot will respond using TF-IDF-based similarity."
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Initialize / Retrieve Chat History
# ──────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []  # Will store tuples: (role, message)

# ──────────────────────────────────────────────────────────────────────────────
# 5) User Input Form
# ──────────────────────────────────────────────────────────────────────────────

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your message here...")
    submit_button = st.form_submit_button(label="Send")
    if submit_button and user_input.strip():
        # Append user message
        st.session_state.history.append(("user", user_input))

        # Get bot response and append
        bot_response = bot.get_answer(user_input)
        st.session_state.history.append(("bot", bot_response))

# ──────────────────────────────────────────────────────────────────────────────
# 6) Display Chat History
# ──────────────────────────────────────────────────────────────────────────────

for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and TF-IDF")
