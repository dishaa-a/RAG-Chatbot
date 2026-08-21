import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- Page config must be the FIRST Streamlit command ---
st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")

# --- Setup: load token safely (local .env OR Streamlit Cloud secrets) ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except Exception:
        HF_TOKEN = None

if not HF_TOKEN:
    st.error("HF_TOKEN not found. Add it to your .env file (local) or Streamlit secrets (deployed).")
    st.stop()

try:
    login(token=HF_TOKEN)
except Exception as e:
    st.error(f"Hugging Face login failed: {e}")
    st.stop()

st.title("🎥 YouTube Transcript RAG Chatbot")

# --- Session state ---
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: video input ---
with st.sidebar:
    st.header("Load a video")
    video_id = st.text_input("YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
    load_btn = st.button("Load Transcript")

def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript_obj = transcript_list.find_transcript(["hi"])
        except NoTranscriptFound:
            available = list(transcript_list)
            transcript_obj = available[0].translate("en")
        fetched = transcript_obj.fetch()
        return " ".join(chunk.text for chunk in fetched)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

@st.cache_resource(show_spinner=False)
def build_chain(_video_id, transcript_text, _hf_token):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=_hf_token,
        max_new_tokens=512,
        temperature=0.3
    )
    llm = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""Answer ONLY using the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    return parallel_chain | prompt | llm | StrOutputParser()

if load_btn and video_id:
    with st.spinner("Fetching transcript and building index..."):
        transcript_text = get_transcript(video_id)
        if transcript_text is None:
            st.error("No transcript available for this video.")
        else:
            try:
                st.session_state.chain = build_chain(video_id, transcript_text, HF_TOKEN)
                st.session_state.messages = []
                st.success("Transcript loaded! Ask away below.")
            except Exception as e:
                st.error(f"Failed to build the RAG chain: {e}")

# --- Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.chain:
    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Load a video from the sidebar to start chatting.")