import streamlit as st
import pdfplumber
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Chunk the text
def chunk_text(text, max_words=150):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Load models and build index
@st.cache_resource
def load_resources():
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    qa = pipeline("text2text-generation", model="google/flan-t5-small", max_length=200)

    chunks = []
    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join("data", filename))
            chunks.extend(chunk_text(text))

    # Filter junk chunks
    chunks = [ch for ch in chunks if len(ch.split()) > 20 and "CURRICULUM" not in ch.upper()]

    # Embed and index
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return model, qa, index, chunks

# UI
st.set_page_config(page_title="Smart College Chatbot")
st.title("📘 Smart College Chatbot")
st.markdown("Ask me anything about your syllabus or college curriculum!")

query = st.text_input("🔎 Your Question:")

if query:
    model, qa, index, chunks = load_resources()
    q_embed = model.encode([query])
    _, I = index.search(np.array(q_embed), k=5)  # ← using more chunks

    # Build context
    context = " ".join([chunks[i] for i in I[0]])
    context = " ".join(context.split()[:450])  # Truncate for safety

    # Improved prompt
    prompt = f"""You are a helpful academic assistant at an Indian engineering college.
Extract and list only the actual subjects mentioned in the following context.
Avoid repeating headers or titles.

Context:
{context}

Question: {query}
"""

    with st.spinner("Thinking..."):
        result = qa(prompt)
        st.success("✅ Answer:")
        st.write(result[0]['generated_text'])
