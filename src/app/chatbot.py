# src/app/chatbot.py
import os
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

# Now we can import our modules
from src.llm.model_config import LLMConfig
from src.utils.config import Config

class MediMindsChat:
    def __init__(self):
        self.config = Config.load_config()
        self.llm_config = LLMConfig()
        self.setup_streamlit()

    def setup_streamlit(self):
        """Setup Streamlit UI"""
        st.title("MediMinds: Your Medical Knowledge Navigator 🧠💊")
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def display_chat_history(self):
        """Display chat history"""
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

    def handle_user_input(self, prompt):
        """Process user input and generate response"""
        try:
            qa_chain = self.llm_config.create_qa_chain()
            response = qa_chain.invoke({'query': prompt})
            return response["result"]  # Return only the result, not the sources
            
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            return None

    def run(self):
        """Main chat application loop"""
        self.display_chat_history()
        
        if prompt := st.chat_input("What would you like to know about?"):
            # Display user message
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({
                'role': 'user', 
                'content': prompt
            })
            
            # Generate and display response
            if response := self.handle_user_input(prompt):
                st.chat_message('assistant').markdown(response)
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': response
                })

if __name__ == "__main__":
    chat_app = MediMindsChat()
    chat_app.run()