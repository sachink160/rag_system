import os
import tempfile
import streamlit as st
# from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


# from langchain.chains import RetrievalQA

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def process_documents(documents, chunk_size=1000):
    """Process documents and create vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def handle_query(query):
    """Handle user query and return response"""
    if st.session_state.vector_store:
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever()
        )
        result = qa.run(query)
        return result
    return "Please upload and process documents first"

def main():
    st.title("📄 Document QA AI Agent")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "csv"],
            accept_multiple_files=True
        )
        
        # Chunk size slider
        chunk_size = st.number_input(
            "Chunk Size (tokens)",
            min_value=500,
            max_value=2000,
            value=1000
        )
        
        # Process button
        if st.button("Process Documents"):
            if uploaded_files:
                documents = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(file.read())
                        file_path = temp_file.name
                        
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                    elif file.name.endswith(".csv"):
                        loader = CSVLoader(file_path, encoding="ISO-8859-1")  # Alternative
                        documents.extend(loader.load())
                    
                    st.session_state.processed_files.append(file.name)
                
                st.session_state.vector_store = process_documents(documents, chunk_size)
                st.success("Documents processed successfully!")
        
        # Delete documents
        if st.button("Delete All Documents"):
            st.session_state.processed_files = []
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.success("All documents and history cleared!")
        
        # Show processed files
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    # Main chat interface
    st.header("Chat with Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle new query
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display AI response
        with st.chat_message("assistant"):
            response = handle_query(prompt)
            st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()