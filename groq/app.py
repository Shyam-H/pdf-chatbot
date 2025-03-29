import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ========== Initialize NeMo Guardrails ==========
config = RailsConfig.from_path("./config.yml")
guardrails = LLMRails(config)

# ========== Streamlit UI ==========
st.set_page_config(
    page_title="🔒 Guarded PDF Chatbot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button { background-color: #4f46e5; }
    .stTextInput input { border-radius: 8px; }
    .assistant-message { background: #f0f0f0; border-radius: 10px; padding: 10px; }
    .user-message { background: #e6f3ff; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.title("Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY")
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

# ========== Document Processing ==========
def process_docs():
    if not uploaded_files:
        st.warning("Upload PDFs first!")
        return
    
    with st.status("Processing..."):
        # Extract text
        text = ""
        for file in uploaded_files:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)

        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Initialize LLM with Guardrails
        llm = ChatGroq(
            groq_api_key = st.secrets["GROQ_API_KEY"],
            model_name="llama3-70b-8192",
            temperature=0.3
        )

        # Conversation chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        st.success("Ready!")

# ========== Chat Interface ==========
st.title("🔒 Guarded PDF Chatbot")

if uploaded_files and groq_api_key:
    process_docs()

if "conversation" in st.session_state:
    # Display history
    for msg in st.session_state.get("chat_history", []):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # User input
    if prompt := st.chat_input("Ask about your docs..."):
        # Apply NeMo Guardrails
        guarded_prompt = guardrails.generate(prompt=prompt)
        
        if guarded_prompt == "blocked":
            st.error("This query was blocked by safety filters.")
        else:
            # Get AI response
            response = st.session_state.conversation({"question": guarded_prompt})
            guarded_response = guardrails.generate(prompt=response["answer"])
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": guarded_response})
            
            st.rerun()
