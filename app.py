import os
from dotenv import load_dotenv
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import hashlib
import requests
from bs4 import BeautifulSoup
import re

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 


def message_bubble(sender, message, is_user=False):
    align = 'flex-end' if is_user else 'flex-start'
    bg_color = '#4F8BF9' if is_user else '#E8F0FE'
    text_color = 'white' if is_user else 'black'
    st.markdown(f'''
    <div style="display: flex; justify-content: {align}; padding: 4px 0;">
        <div style="background-color: {bg_color}; color: {text_color}; padding: 10px 15px; border-radius: 12px; max-width: 80%;">
            <b>{sender}</b><br>{message}
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Ask Gemini
def ask_gemini(context, question):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    prompt = f"""Use the context below to answer the question in detail.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text

# Search online references
def search_online_reference(question):
    search_url = f"https://www.google.com/search?q={question.replace(' ', '+')}+site:researchgate.net+OR+site:arxiv.org+OR+site:ieee.org"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for g in soup.find_all('a'):
        href = g.get('href')
        if href and ("arxiv.org" in href or "ieee.org" in href or "researchgate.net" in href):
            links.append(href)
    return links[:3]

# Extract arXiv ID
def extract_arxiv_id(query):
    match = re.search(r'arXiv:(\d+\.\d+)', query)
    return match.group(1) if match else None

# Download arXiv paper
def download_arxiv_pdf(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    if response.status_code == 200:
        pdf_path = f"{arxiv_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return pdf_path
    return None

# Handle PDF
def handle_pdfs(uploaded_pdfs):
    full_text = ""
    for pdf in uploaded_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                full_text += content
    return full_text

# Main app
def main():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            .chat-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                background-color: #f9f9f9;
                border-radius: 20px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            }
            .chat-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
            }
            .upload-btn {
                background-color: #1f4ef4;
                color: white;
                padding: 0.6rem 1.2rem;
                font-size: 1rem;
                border-radius: 10px;
                border: none;
                margin-top: 1rem;
            }
            .bubble {
                max-width: 80%;
                padding: 1rem;
                margin: 0.5rem;
                border-radius: 15px;
                font-size: 1.05rem;
                line-height: 1.5;
            }
            .bot-bubble {
                background-color: #e0e0e0;
                align-self: flex-start;
            }
            .user-bubble {
                background-color: #c7f0d8;
                align-self: flex-end;
            }
        </style>
        <div class='chat-container'>
            <div class='chat-icon'>🤖</div>
            <h2>Upload a file to get started</h2>
        </div>
    """, unsafe_allow_html=True)

    uploaded_pdfs = st.file_uploader("Upload File", type='pdf', accept_multiple_files=True)

    st.markdown("""
        <div class='bubble bot-bubble'>Hello! How can I assist you today?</div>
    """, unsafe_allow_html=True)

    query = st.text_input("Enter your question...")
    send = st.button("Send")

    if send and query:
        arxiv_id = extract_arxiv_id(query)
        context = ""

        if uploaded_pdfs:
            file_names = "".join(sorted([pdf.name for pdf in uploaded_pdfs]))
            store_hash = hashlib.md5(file_names.encode()).hexdigest()
            store_name = f"vectorstore_{store_hash}"

            if "vectorstore" not in st.session_state or st.session_state.get("store_name") != store_name:
                with st.spinner("Reading and extracting text from PDFs..."):
                    full_text = handle_pdfs(uploaded_pdfs)

                with st.spinner("Generating or loading vector embeddings..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(full_text)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    if os.path.exists(f"{store_name}.pkl"):
                        with open(f"{store_name}.pkl", "rb") as f:
                            vectorstore = pickle.load(f)
                    else:
                        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                        with open(f"{store_name}.pkl", "wb") as f:
                            pickle.dump(vectorstore, f)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.store_name = store_name

            vectorstore = st.session_state.vectorstore
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([" ".join(doc.page_content.split()) for doc in docs])
            if not context.strip() or len(context.strip()) < 100:
                st.info("No strong context found in PDFs. Searching online instead...")
                links = search_online_reference(query)
                context = ""
                for link in links:
                    if "arxiv.org/abs/" in link:
                        match = re.search(r'arxiv.org/abs/(\d+\.\d+)', link)
                        if match:
                            arxiv_id = match.group(1)
                            st.info(f"Fetching reference from arXiv ID {arxiv_id}...")
                            pdf_path = download_arxiv_pdf(arxiv_id)
                            if pdf_path:
                                full_text = handle_pdfs([pdf_path])
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                                chunks = text_splitter.split_text(full_text)
                                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                                temp_vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                                docs = temp_vectorstore.similarity_search(query, k=3)
                                context = "\n\n".join([" ".join(doc.page_content.split()) for doc in docs])
                            break
                if not context:
                    context = "".join([f"Found: {link}\n" for link in links]) or "No relevant references found online."

        elif arxiv_id:
            with st.spinner(f"Downloading and processing arXiv paper {arxiv_id}..."):
                pdf_path = download_arxiv_pdf(arxiv_id)
                if pdf_path:
                    full_text = handle_pdfs([pdf_path])
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(full_text)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    temp_vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                    docs = temp_vectorstore.similarity_search(query, k=3)
                    context = "\n\n".join([" ".join(doc.page_content.split()) for doc in docs])
                else:
                    context = f"Failed to fetch arXiv paper for ID {arxiv_id}."

        else:
            st.info("No PDFs uploaded. Searching online using citation clues...")
            links = search_online_reference(query)
            context = ""
            for link in links:
                if "arxiv.org/abs/" in link:
                    match = re.search(r'arxiv.org/abs/(\d+\.\d+)', link)
                    if match:
                        arxiv_id = match.group(1)
                        st.info(f"Fetching reference from arXiv ID {arxiv_id}...")
                        pdf_path = download_arxiv_pdf(arxiv_id)
                        if pdf_path:
                            full_text = handle_pdfs([pdf_path])
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            chunks = text_splitter.split_text(full_text)
                            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            temp_vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                            docs = temp_vectorstore.similarity_search(query, k=3)
                            context = "\n\n".join([" ".join(doc.page_content.split()) for doc in docs])
                        break
            if not context:
                context = "".join([f"Found: {link}\n" for link in links]) or "No relevant references found online."

        response = ask_gemini(context, query)
        if "context_history" not in st.session_state:
            st.session_state.context_history = []
        st.session_state.context_history.append(("You", query))
        st.session_state.context_history.append(("Gemini", response))

    # Chat History Rendering
    if "context_history" in st.session_state:
        for sender, message in st.session_state.context_history:
            is_user = sender == "You"
            message_bubble(sender, message, is_user=is_user)

if __name__ == "__main__":
    main()
