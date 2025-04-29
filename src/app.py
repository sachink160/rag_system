import streamlit as st


def main():
    st.set_page_config(page_title="Chat with mutiple pdf", page_icon=":books:")
    
    st.header("Chat with multiple PDFs")
    st.text_input("Ask Your question about your documents: ")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your files here...", accept_multiple_files=True)
        st.button("Process")
if __name__ == "__main__":
    main()