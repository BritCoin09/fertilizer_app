import streamlit as st
import faiss
import numpy as np
import openai
import glob
import os

# -----------------------------------
# 🌿 OpenAI API Key
# -----------------------------------
openai.api_key = ("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")

# -----------------------------------
# 🌿 Page Config
# -----------------------------------
st.set_page_config(
    page_title="Fertilizer Intelligence Assistant",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------------
# 🌾 Sidebar
# -----------------------------------
st.sidebar.markdown("""
# 🌱 Fertilizer Intelligence Assistant
Ask questions about:
- Nutrients (N, P, K)
- Soil health
- Crop responses
- Fertilizer strategies
- Your transcripts + notes
""")

query = st.sidebar.text_input("Ask a question:")

# -----------------------------------
# 📦 Load Documents
# -----------------------------------
@st.cache_data
def load_documents():
    docs = []
    for path in glob.glob("text/*.txt*"):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"path": path, "text": f.read()})
    for path in glob.glob("transcripts/*.txt*"):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"path": path, "text": f.read()})
    return docs

documents = load_documents()

# -----------------------------------
# ✂️ Chunking
# -----------------------------------
def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = []
for d in documents:
    for c in chunk_text(d["text"]):
        chunks.append({"text": c, "source": d["path"]})

# 🧠 Load FAISS index + embeddings
# -----------------------------------
import pickle

@st.cache_resource
def load_faiss_index():
    # Load embeddings
    with open("vectorstore/fertilizer_index/index.pkl", "rb") as f:
        embeddings = pickle.load(f)

    embs_np = np.array(embeddings).astype("float32")

    # Load FAISS index
    index = faiss.read_index("vectorstore/fertilizer_index/index.faiss")

    return index, embs_np

index, embs_np = load_faiss_index()


# -----------------------------------
# 🔍 Retrieval
# -----------------------------------
def retrieve(query, k=5):
    q_emb = openai.Embedding.create(
        model="text-embedding-3-small",
        input=[query]
    )["data"][0]["embedding"]

    q_emb = np.array(q_emb).astype("float32").reshape(1, -1)
    D, I = index.search(q_emb, k)
    return [chunks[idx] for idx in I[0]]

# -----------------------------------
# 💬 Answer (old OpenAI syntax)
# -----------------------------------
def answer_question(q):
    retrieved = retrieve(q)
    context = "\n\n---\n\n".join(
        [f"Source: {d['source']}\n\n{d['text']}" for d in retrieved]
    )

    prompt = (
        "Use the following context to answer the question. "
        "If the context is partially relevant, infer the best answer. "
        "If unrelated, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely:"
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return resp["choices"][0]["message"]["content"].strip(), retrieved

# -----------------------------------
# 🌻 Main Page
# -----------------------------------
st.title("🌿 Fertilizer Intelligence Dashboard")

st.markdown("""
Welcome to your **Fertilizer Intelligence Assistant** — a retrieval‑augmented system built from your notes, transcripts, and fertilizer research.

Use the sidebar to ask questions anytime.
""")

# -----------------------------------
# 🌾 Answer Display
# -----------------------------------
if query:
    st.subheader("🌱 Answer")
    answer, retrieved_chunks = answer_question(query)
    st.write(answer)

    st.subheader("📄 Retrieved Context")
    for c in retrieved_chunks:
        with st.expander(c["source"]):
            st.write(c["text"])

