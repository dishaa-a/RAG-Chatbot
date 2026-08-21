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
RAG Chatbot/
├── app.py # Streamlit application (main entry point)
├── requirements.txt # Python dependencies
├── .env # Local environment variables (not committed)
├── .gitignore
└── README.md


## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/dishaa-a/Gen_AI.git
cd Gen_AI
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:

HF_TOKEN=your_huggingface_token_here

Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## Usage

1. Paste a YouTube video ID into the sidebar (the part after `v=` in the video URL).
2. Click **Load Transcript**.
3. Ask questions about the video in the chat box.
4. The chatbot answers using only the video's transcript content.

## Notes

- Some Hugging Face models require accepting a license on their model page before use (e.g., Meta Llama models).
- If a video has no captions available, the app will notify you that no transcript could be loaded.
- The Hugging Face API token should never be committed to version control — it is loaded via `.env` locally or via Streamlit secrets in deployment.

## Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own version:
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set `HF_TOKEN` under the app's **Secrets** settings.
4. Deploy.

## Future Improvements

- [ ] Multi-turn conversation memory
- [ ] Source citation with transcript timestamps
- [ ] Support for multiple videos / playlists
- [ ] Answer evaluation and quality metrics

## License

This project is open source and available under the [MIT License](LICENSE).

