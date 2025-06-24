import streamlit as st
import pdfplumber
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Function to extract all text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Split the full text into small overlapping chunks
def chunk_text(text, max_words=150):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Cache the model and index to avoid reloading
@st.cache_resource
def load_resources():
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    qa = pipeline("text2text-generation", model="google/flan-t5-small", max_length=200)
    chunks = []

    for filename in os.listdir("data"):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join("data", filename))
            chunks.extend(chunk_text(text))

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return model, qa, index, chunks

# Streamlit UI
st.set_page_config(page_title="Smart College Chatbot")
st.title("📘 Smart College Chatbot")
st.markdown("Ask me anything about your syllabus or college curriculum!")

# User input
query = st.text_input("🔎 Your Question:")

if query:
    model, qa, index, chunks = load_resources()
    q_embed = model.encode([query])
    _, I = index.search(np.array(q_embed), k=3)

    # Build the combined context
    context = " ".join([chunks[i] for i in I[0]])
    context = " ".join(context.split()[:450])  # Limit word count for safety

    # Craft a more helpful prompt
    prompt = f"""You are a helpful college assistant.
Using only the below context, answer the question clearly and briefly.
Give only relevant subject names, semester info, or curriculum details.

Context:
{context}

Question: {query}
"""

    with st.spinner("Thinking..."):
        result = qa(prompt)
        st.success("✅ Answer:")
        st.write(result[0]['generated_text'])
