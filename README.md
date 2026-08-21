# 🎥 YouTube Transcript RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you chat with any YouTube video by asking questions about its content. Built with LangChain, Hugging Face models, and Streamlit.

## How It Works

1. **Transcript Extraction** — Fetches the transcript of a YouTube video using its video ID.
2. **Chunking** — Splits the transcript into overlapping text chunks for better context retrieval.
3. **Embedding & Storage** — Converts chunks into vector embeddings (Hugging Face `sentence-transformers`) and stores them in a Chroma vector database.
4. **Retrieval** — On each question, retrieves the most relevant transcript chunks using similarity search.
5. **Generation** — Passes the retrieved context + question to a Hugging Face LLM to generate a grounded answer.
6. **Chat UI** — A Streamlit interface lets you load a video and chat with it interactively.

## Tech Stack

- **LangChain** — orchestration of the RAG pipeline
- **Hugging Face** — embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and LLM (`Qwen/Qwen2.5-7B-Instruct`)
- **ChromaDB** — vector store for similarity search
- **Streamlit** — web interface
- **youtube-transcript-api** — transcript fetching

## Project Structure