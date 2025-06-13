An interactive Streamlit web application that allows users to upload research papers (PDFs), ask questions, and receive context-aware answers. It leverages Google Gemini, HuggingFace sentence embeddings, LangChain for text chunking, and FAISS for efficient semantic search. Additionally, it searches and uses online references when local context is insufficient.

🚀 Features
📄 Upload multiple research paper PDFs.
❓ Ask natural language questions about the papers.
🧠 Uses semantic search (FAISS + HuggingFace Embeddings) for contextual retrieval.
🤖 Answers are generated using Google Gemini Pro.
🌐 If context is insufficient, it searches scholarly sources like arXiv, IEEE, and ResearchGate online.
💬 Clean chat UI with styled message bubbles.
🔒 Session-aware vectorstore caching using hashing for efficient reuse.

🧰 Tech Stack
Component	Technology
Frontend/UI	Streamlit
Embeddings	HuggingFace - MiniLM-L6-v2
Vector Store	FAISS
Text Chunking	LangChain
PDF Parsing	PyPDF2
Web Scraping	requests + BeautifulSoup
LLM Integration	Google Gemini API
