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

# === Setup OpenAI Chat LLM ===
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=2000,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_key
)

# === Setup RAG Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff"
)

# === Streamlit UI Setup ===
st.set_page_config(page_title="United Methodist Church Assistant", layout="wide")
st.image("UMC_LOGO.png", width=120)
st.title("United Methodist Church Assistant")
st.markdown("Ask a question about the Book of Doctrines & Discipline.")

# === Initialize session state ===
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = None

# === Query Input ===
query = st.text_input("🔎 Enter your question:", value=st.session_state.query)

# === Process New Query ===
if st.button("Get Answer") and query.strip():
    with st.spinner("Generating response..."):
        response = qa_chain.invoke(query)
        st.session_state.query = query
        st.session_state.response = response
        st.success("✅ Response generated")

# === Display Stored Response if Available ===
if st.session_state.response:
    resp = st.session_state.response
    q_text = st.session_state.query

    st.markdown("### 💬 Response")
    if isinstance(resp, dict):
        st.markdown(f"<b>Question:</b> {q_text}", unsafe_allow_html=True)
        st.markdown(f"""<div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
        {resp['result']}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
        {resp}
        </div>""", unsafe_allow_html=True)

    # === Toggle to Show Source Documents ===
    if st.checkbox("📄 Show Retrieved Source Documents"):
        retrieved_docs = vector_store.similarity_search(q_text, k=5)
        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata
            st.markdown(f"**Document {i+1}:**")
            st.markdown(f"- **Part:** {meta.get('part', 'N/A')}")
            st.markdown(f"- **Section:** {meta.get('section_title', 'N/A')}")
            st.markdown(f"- **Paragraph #:** {meta.get('paragraph_number', 'N/A')}")
            st.markdown(f"- **Para Title:** {meta.get('para_title', 'N/A')}")
            st.markdown(f"- **Sub Para Title:** {meta.get('sub_para_title', 'N/A')}")
            st.markdown(f"**Content:** {doc.page_content}")
            st.markdown("---")
