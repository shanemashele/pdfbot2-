import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import io
import PyPDF2
import time

load_dotenv()

# Load the API keys
groq_api_key = os.getenv('GROQ_API_KEY', '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')
google_api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDVPVSB2NZtL4zA4I-1eGQ2tjLtqoRFhNc')

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

def vector_embedding(pdf_file):
    # Check if Google API Key is available
    if not google_api_key:
        st.error("Google API key is missing. Please set the GOOGLE_API_KEY environment variable.")
        return

    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key,  # Ensure the key is passed if required
        model="models/embedding-001"
    )

    # Load PDF and create document chunks
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Create Document objects
    documents = [Document(text, metadata={"source": "PDF"})]
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Upload PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf is not None:
    vector_embedding(uploaded_pdf)
    st.write("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Ask Question"):
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")
