import streamlit as st
import faiss
import json
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

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
    max_tokens=2500,
    model_name="gpt-4o-mini",
    openai_api_key=openai_key
)

# === Streamlit UI Setup ===
st.set_page_config(page_title="United Methodist Church Assistant", layout="wide")
st.image("UMC_LOGO.png", width=80)
st.title("United Methodist Church Assistant")
st.markdown("Ask a question about the Book of Doctrines & Discipline.")

# === Initialize chat history state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Start New Chat Button ===
if st.button("🧹 Start New Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# === Display chat history ===
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"""<div style="background-color: #f1f1f1; padding: 12px; border-radius: 5px; border: 1px solid #ccc;">
    {chat['answer']}
    </div>""", unsafe_allow_html=True)

# === Input for new question ===
query = st.text_input("🔎 Enter your question:", "")

if st.button("Send") and query.strip():
    with st.spinner("Thinking..."):
        # === Retrieve top 5 docs ===
        retrieved_docs = vector_store.similarity_search(query, k=5)

        # === Build context from last 5 Q&A ===
        history_context = ""
        for qna in st.session_state.chat_history[-5:]:
            history_context += f"Q: {qna['question']}\nA: {qna['answer']}\n\n"

        # === Custom Prompt Template with Chat History ===
        prompt_template = """You are an expert assistant on the Book of Doctrines & Discipline of the United Methodist Church.

Use the previous Q&A and reference documents below to answer the user's question in a detailed, well-structured, and professional tone.

### Previous Q&A (for context):
{chat_history}

### Reference Documents:
{context}

### Question:
{question}

### Answer:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=prompt_template
        )

        # === Prepare StuffDocumentsChain ===
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        # === Run chain with chat context + docs ===
        response = stuff_chain.run(
            input_documents=retrieved_docs,
            chat_history=history_context,
            question=query
        )

        # === Append new Q&A to session state ===
        st.session_state.chat_history.append({"question": query, "answer": response})

        # === Display response ===
        st.markdown(f"**You:** {query}")
        st.markdown(f"""<div style="background-color: #f1f1f1; padding: 12px; border-radius: 5px; border: 1px solid #ccc;">
        {response}
        </div>""", unsafe_allow_html=True)

        # === Toggle for Retrieved Source Documents ===
        if st.checkbox("📄 Show Retrieved Source Documents"):
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
