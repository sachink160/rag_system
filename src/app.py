#V2_code

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Dummy credentials (replace with secure method in production)
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "password"
}

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def login():
    """Simple login form"""
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")


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


def main_app():
    """Main Document QA App after login"""
    st.title("📄 Document QA AI Agent")

    # Sidebar for document management
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}")
        
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
                        loader = CSVLoader(file_path, encoding="ISO-8859-1")
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


def main():
    if not st.session_state.authenticated:
        login()
    else:
        main_app()


if __name__ == "__main__":
    main()



#v1_code
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, OpenAI
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from dotenv import load_dotenv

# load_dotenv()


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def vectorestore_creator(chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm =  OpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
    
#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write((message.content), unsafe_allow_html=True)
#         else:
#             st.write((message.content), unsafe_allow_html=True)


# def main():
#     st.set_page_config(page_title="Chat with mutiple pdf", page_icon=":books:")
    
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
    
#     st.header("Chat with multiple PDFs")
#     user_question = st.text_input("Ask Your question about your documents: ")
    
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader("Upload your files here...", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing..."):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)
            
#                 # get text chunks from pdf
#                 raw_chunks = get_text_chunks(raw_text)
            
#                 # create vesctor store
#                 vectore_store = vectorestore_creator(raw_chunks)
                
#                 # Create conversation chain
#                 convestion = get_conversation_chain(vectore_store)
                
#                 st.session_state.conversation = convestion

# if __name__ == "__main__":
#     main()