import streamlit as st
import pdfplumber
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load PDF files and extract text
def load_texts_from_pdfs(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                chunks = chunk_text(full_text)
                all_chunks.extend(chunks)
    return all_chunks

def chunk_text(text, max_words=150):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

@st.cache_resource
def setup():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline("text2text-generation", model="google/flan-t5-base")
    chunks = load_texts_from_pdfs("data")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return model, qa, index, chunks

st.title("📘 Smart College Chatbot")
st.write("Ask me anything about your syllabus or college curriculum!")

query = st.text_input("🔎 Your Question:")
if query:
    model, qa, index, chunks = setup()
    q_embed = model.encode([query])
    _, I = index.search(np.array(q_embed), k=3)
    context = " ".join([chunks[i] for i in I[0]])
    prompt = f"Answer this using the context:\n{context}\nQuestion: {query}"
    with st.spinner("Thinking..."):
        answer = qa(prompt)[0]['generated_text']
    st.success("✅ Answer:")
    st.write(answer)
