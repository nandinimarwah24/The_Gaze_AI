# 🧠 The Gaze AI

> **An offline AI-powered Retrieval-Augmented Generation (RAG) chatbot built using FAISS, Sentence Transformers, Ollama, and Streamlit.**

The Gaze AI is a privacy-focused document assistant that enables users to chat with their PDF documents entirely on their local machine. It combines semantic search with Large Language Models (LLMs) to generate accurate, context-aware responses while keeping all data offline.

---

## ✨ Features

* 📄 Chat with one or multiple PDF documents
* 🔍 Semantic document retrieval using FAISS Vector Search
* 🤖 Local LLM inference with Ollama
* 🧠 Short-term conversational memory
* 📚 Source citation with PDF name and page number
* ⚡ Fast embedding generation using Sentence Transformers
* 🎨 Modern Streamlit interface
* 🔒 Fully offline — no cloud APIs required

---

## 🏗️ Architecture

```
                PDF Documents
                      │
                      ▼
              Text Extraction
                      │
                      ▼
              Text Chunking
                      │
                      ▼
       Sentence Transformer Embeddings
                      │
                      ▼
               FAISS Vector Index
                      │
      User Question ──┘
                      │
                      ▼
          Similar Chunk Retrieval
                      │
                      ▼
            Ollama Local LLM
                      │
                      ▼
          Grounded AI Response
```

---

## 📂 Project Structure

```
The-Gaze-AI/
│
├── data/
│   ├── *.pdf
│   ├── chunks.json
│   ├── embeddings.npy
│   └── faiss.index
│
├── app.py                 # Streamlit application
├── rag.py                 # PDF ingestion & indexing
├── query.py               # CLI chatbot
├── local_llm.py           # Ollama interface
├── faiss_store.py         # Vector database utilities
├── requirement.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Ollama
* NumPy
* PyPDF
* Transformers

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/nandinimarwah24/The_Gaze_AI.git
cd the-gaze-ai
```

Create a virtual environment.

```bash
python -m venv rag_env
source rag_env/bin/activate      # macOS/Linux

# Windows
rag_env\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirement.txt
```

---

## 🤖 Install Ollama

Download and install Ollama.

```bash
https://ollama.com/download
```

Pull the default model.

```bash
ollama pull llama3.2:3b
```

Start the Ollama server.

```bash
ollama serve
```

---

## 📚 Build the Knowledge Base

Place your PDF documents inside the `data/` directory.

Run the indexing pipeline.

```bash
python rag.py
```

Generated files:

* `chunks.json` – processed document chunks
* `embeddings.npy` – embedding vectors
* `faiss.index` – FAISS vector database

---

## 💬 Command Line Chat

Ask a single question.

```bash
python query.py "What is phishing?"
```

Retrieve more relevant chunks.

```bash
python query.py --top-k 5 "Explain asymmetric encryption."
```

Start an interactive chat session.

```bash
python query.py
```

Available commands:

```
/history   View conversation history
/clear     Clear memory
exit       Exit chatbot
```

Customize memory size.

```bash
python query.py --memory-turns 3
```

Display retrieved source chunks.

```bash
python query.py --show-sources
```

Use a different Ollama model.

```bash
python query.py --llm-model mistral
```

---

## 🌐 Streamlit Web Interface

Launch the application.

```bash
streamlit run app.py
```

The web interface allows you to:

* Upload one or multiple PDFs
* Build a local knowledge base
* Chat with your documents
* View cited document sources
* Maintain short-term conversational context during the session

---

## 🧠 Memory

The Gaze AI currently implements **short-term conversational memory**.

* Stores recent conversation history
* Improves follow-up questions
* Exists only during the active session
* Automatically clears when the application closes

---

## 🔒 Privacy

All processing is performed locally.

* No cloud APIs
* No external vector databases
* No document uploads
* Complete offline inference using Ollama

Your documents never leave your computer.

---

## 📌 Future Improvements

* Long-term memory
* Hybrid keyword + semantic retrieval
* Conversation summarization
* OCR support for scanned PDFs
* Multi-modal document understanding
* Document collections and workspaces
* Authentication and user profiles
* Support for multiple embedding models
* Response streaming

---

## 📄 License

This project is available under the MIT License.

---

## 👩‍💻 Author

**Nandini Marwah**

BCA Student • Cybersecurity Enthusiast • AI Developer

Building privacy-first AI applications powered by local LLMs.
