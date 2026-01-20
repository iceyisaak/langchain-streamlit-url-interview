# 🤖 RAG Chatbot: URL Interview

StreamlitUI: https://langchain-app-url-interview.streamlit.app/

Repo: https://github.com/iceyisaak/langchain-streamlit-url-interview

---

A Retrieval-Augmented Generation (RAG) conversational agent that allows users to "interview" web content. By providing a URL (YouTube video or Website), the chatbot scrapes the content, indexes it into a vector store, and provides context-aware answers to user queries using **Groq** and **LangChain**.

## ✨ Features

* **Multi-Source Loading**: Supports both YouTube video transcripts and general website content.
* **Contextual Memory**: Remembers past interactions within a session to handle follow-up questions effectively.
* **High Performance**: Powered by **Llama 3.1 (8B)** on Groq for near-instantaneous inference.
* **Vector Search**: Uses **ChromaDB** and **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for accurate document retrieval.
* **Multilingual Support**: Automatically responds in the language used by the user.

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **LLM**: Groq (Llama-3.1-8b-instant)
* **Embeddings**: HuggingFace (Open-source)
* **Vector Database**: ChromaDB

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed. You will also need:

* A **Groq API Key** (Get it at [Groq Cloud](https://console.groq.com/))
* A **HuggingFace Token** (Get it at [HuggingFace Settings](https://huggingface.co/settings/tokens))

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/iceyisaak/langchain-streamlit-url-interview.git
cd langchain-streamlit-url-interview
pip install -r requirements.txt

```

### 3. Dependencies

Ensure your `requirements.txt` includes:

```text
streamlit
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-chroma
validators
unstructured
youtube-transcript-api
pytube
python-dotenv

```

### 4. Running the App

```bash
streamlit run app.py

```

---

## 💡 How to Use

1. **Configure API Keys**: Enter your Groq and HuggingFace tokens in the sidebar.
2. **Input URL**: Paste a link to a YouTube video or a website article.
3. **Wait for Processing**: The app will load, split, and embed the content into a temporary vector store.
4. **Chat**: Ask questions like "What are the main takeaways?" or "Summarize the third paragraph."

---

## 🏗️ System Architecture

The app follows a standard RAG pipeline:

1. **Load**: Extract text using `YoutubeLoader` or `UnstructuredURLLoader`.
2. **Split**: Break text into 5000-character chunks with `RecursiveCharacterTextSplitter`.
3. **Embed**: Convert chunks into vectors using HuggingFace.
4. **Retrieve**: When a user asks a question, the system finds the 5 most relevant chunks.
5. **Generate**: The LLM synthesizes an answer based solely on the retrieved context and chat history.

---

## ⚠️ Important Notes

* **Session Management**: Use the "Session ID" field to maintain different conversation threads.
* **Language Persistence**: The bot is prompted to match the user's input language.
* **Conciseness**: The system is hardcoded to provide answers within a maximum of 3 sentences for efficiency.

---