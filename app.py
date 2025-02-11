import streamlit as st   # For GUI purpose
from PyPDF2 import PdfReader #For extracting text from pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter # It is used for breaking down larger docs into small chunks
# (respects natural boundaries)

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For generating embeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS #For storing embeddings locally and for similarity search
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain  #For generating precise and contextually accurate froms extracted chunks
from langchain.prompts import PromptTemplate #To create customised prompt
from dotenv import load_dotenv #for loading environment variables
import base64 #for displaying pdf
from typing import List, Dict

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)

        # Extracting text from pdf

        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# For generating and storing the embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def format_chat_history(chat_history: List[Dict]) -> str: #chat_history: A list of dictionaries where each dictionary contains two keys question and answer

    formatted_history = ""

    for message in chat_history:
        formatted_history += f"Human: {message['question']}\n"
        formatted_history += f"Assistant: {message['answer']}\n"
    return formatted_history

def get_conversational_chain():
    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided context and previous conversation history.
    Use the conversation history to maintain context and provide more relevant answers.
    If the answer cannot be found in the context or conversation history, just say "answer is not available in the context."
    Don't provide incorrect information.

    Previous conversation:
    {chat_history}

    Current context:
    {context}

    Current question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

        # generating embeddings of user_question
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        formatted_history = format_chat_history(chat_history)
        
        response = chain(
            {
                "input_documents": docs,
                "question": user_question,
                "chat_history": formatted_history
            },
            return_only_outputs=True
        )
        
        # Add the new Q&A pair to chat history
        chat_history.append({
            "question": user_question,
            "answer": response["output_text"]
        })
        
        return response["output_text"]
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if "faiss_index" not in os.listdir():
            st.warning("Please upload and process a PDF document first before asking questions.")
        return None

def show_pdf_from_bytes(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=1&scrollbar=1"
            width="100%"
            height="800px"
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_chat_history(chat_history: List[Dict]):
    for message in chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("Chat with PDF")

    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None
    if 'pdf_contents' not in st.session_state:
        st.session_state.pdf_contents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True,
                                  type="pdf")
        
        if st.button("Submit & Process"):
            if pdf_docs:
                # Clear previous contents
                st.session_state.pdf_contents = []
                # Store PDF contents
                for pdf in pdf_docs:
                    bytes_data = pdf.read()
                    st.session_state.pdf_contents.append({
                        'name': pdf.name,
                        'content': bytes_data
                    })
                    pdf.seek(0)
                
                st.session_state.pdf_docs = pdf_docs
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload a PDF file first.")
        
        # Add a clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])

    with col1:
        display_chat_history(st.session_state.chat_history)
        
        user_question = st.chat_input("Ask a question about your PDFs")
        if user_question:
            if os.path.exists("faiss_index"):
                response = user_input(user_question, st.session_state.chat_history)
                if response:
                    st.rerun()  
            else:
                st.warning("Please upload and process a PDF document first before asking questions.")

    with col2:
        # For displaying PDF Viewer Section
        if st.session_state.pdf_contents:
            with st.expander("📄 View PDF Documents", expanded=True):
                # Create tabs for each PDF
                tab_titles = [pdf['name'] for pdf in st.session_state.pdf_contents]
                tabs = st.tabs(tab_titles)
                
                # Display PDFs in tabs
                for tab, pdf_data in zip(tabs, st.session_state.pdf_contents):
                    with tab:
                        show_pdf_from_bytes(pdf_data['content'])

if __name__ == "__main__":
    main()