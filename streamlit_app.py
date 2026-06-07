import os
import streamlit as st
from dotenv import load_dotenv

from agentic_rag import setup_agent as get_agent


load_dotenv()

# --- 1. SET UP THE STREAMLIT UI ---
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="centered")
st.title("🤖 Agentic RAG Assistant")
st.caption("Ask me anything about the uploaded document! I can evaluate my own answers and remember our conversation.")

# --- 2. CACHE THE HEAVY LIFTING ---
# The @st.cache_resource decorator ensures this only runs ONCE when the app starts.
@st.cache_resource
def setup_agent():
    return get_agent()

# --- 3. INITIALIZE APP ---
# This shows a spinning wheel while the PDF is loading for the first time
with st.spinner("Initializing AI Agent and Reading Document..."):
    app = setup_agent()

# Set up the memory banks in Streamlit's Session State
if "chat_ui_messages" not in st.session_state:
    st.session_state.chat_ui_messages = []
    
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []

# --- 4. RENDER PREVIOUS CHAT MESSAGES ---
for msg in st.session_state.chat_ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. HANDLE NEW USER INPUT ---
# This creates the text box at the bottom of the screen
if prompt := st.chat_input("Ask a question about your document..."):
    
    # 1. Show the user's message in the UI immediately
    st.session_state.chat_ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is searching and thinking..."):
            inputs = {
                "question": prompt,
                "chat_history": st.session_state.agent_memory,
                "retry_count": 0
            }
            
            result = app.invoke(inputs)
            answer = result["generation"]
            
            # Show the final answer in the UI
            st.markdown(answer)
            
            # 3. Save the interaction to memory
            st.session_state.chat_ui_messages.append({"role": "assistant", "content": answer})
            st.session_state.agent_memory.append(f"User: {prompt}")
            st.session_state.agent_memory.append(f"Agent: {answer}")
            
            # Keep memory from getting too long (max 8 items)
            if len(st.session_state.agent_memory) > 8:
                st.session_state.agent_memory = st.session_state.agent_memory[-8:]