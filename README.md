# 🧠 Agentic RAG Chatbot

An **Agentic Retrieval-Augmented Generation (RAG)** chatbot built with Python, LangChain, LangGraph, Groq LLM, Hugging Face embeddings, and Chroma vector database.

The application allows users to ask questions about the content of a PDF document. Instead of directly sending the question to an LLM, the system retrieves relevant document chunks and uses an agentic workflow to rewrite queries, evaluate retrieved documents, and generate answers using the retrieved context.

---

## 🚀 Features

- 📄 PDF document loading using `PDFPlumberLoader`
- ✂️ Recursive text chunking
- 🧠 Hugging Face sentence embeddings
- 🔎 Semantic similarity search using Chroma
- 🔄 Query rewriting for better retrieval
- 📝 Document relevance grading
- 🤖 Agentic workflow using LangGraph
- 🔁 Retry mechanism for poor retrieval results
- 💬 Conversation history / short-term memory
- ⚡ Groq LLM inference using Llama 3.3 70B
- 🎯 Context-grounded answer generation
- 🖥️ Streamlit interface

---

# 🏗️ Architecture

The application follows an Agentic RAG pipeline:

```text
                         ┌─────────────────────┐
                         │    PDF Document     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   PDF Loader        │
                         │ PDFPlumberLoader    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Text Splitter      │
                         │ RecursiveCharacter   │
                         │ TextSplitter        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Embeddings      │
                         │ all-MiniLM-L6-v2    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Chroma Vector     │
                         │      Database       │
                         └─────────────────────┘


User Question
      │
      ▼
┌─────────────────────┐
│   Query Rewriting   │
│  LangGraph Node     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Retriever       │
│  Top-K Documents    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Document Relevance  │
│       Grader        │
└──────────┬──────────┘
           │
       ┌───┴────┐
       │        │
 Relevant     Not Relevant
       │        │
       │        ▼
       │   ┌─────────────┐
       │   │ Retry /     │
       │   │ Rewrite     │
       │   └──────┬──────┘
       │          │
       │          └──────► Retrieve
       │
       ▼
┌─────────────────────┐
│ Answer Generation   │
│      using LLM      │
└──────────┬──────────┘
           │
           ▼
      Final Answer
