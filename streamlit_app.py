import streamlit as st
import faiss
import json
import os
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# === Load OpenAI API key from Streamlit secret ===
openai_key = st.secrets["OPENAI_API_KEY"]

# === Load FAISS index ===
faiss_index = faiss.read_index("faiss_index.index")

# === Load metadata ===
with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

# === Convert metadata into LangChain Documents ===
documents = [
    Document(page_content=entry["metadata"]["text"], metadata=entry["metadata"])
    for entry in metadata_list
]

# === Wrap in InMemoryDocstore ===
docstore = InMemoryDocstore({str(i): documents[i] for i in range(len(documents))})

# === Setup OpenAI Embeddings ===
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_key)

# === Setup FAISS VectorStore ===
vector_store = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id={i: str(i) for i in range(len(documents))}
)

# === Setup OpenAI LLM ===
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# === Setup RAG Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff"
)

# === Streamlit UI ===
st.set_page_config(page_title="Global Methodist RAG Bot", layout="wide")
st.title("📘 Global Methodist RAG Assistant")
st.markdown("Ask a question about the Book of Doctrines & Discipline.")

query = st.text_input("🔎 Enter your question:", "")

if st.button("Get Answer") and query.strip():
    with st.spinner("Generating response..."):
        response = qa_chain.invoke(query)
        st.success("✅ Response generated")

        st.markdown("### 💬 Response")
        st.write(response)
