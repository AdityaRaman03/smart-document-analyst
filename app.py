import shutil
import streamlit as st
from ingestion_pipeline import load_documents, text_splitter, create_embeddings
from retrieval_generation import generate_q

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Document Q&A")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of (query, answer, sources)

with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Ingest"):
        with st.spinner("Ingesting..."):
            shutil.rmtree("./chroma_db", ignore_errors=True)
            docs = load_documents(uploaded.getvalue())
            if not docs:
                st.error("Please input a valid text-based PDF")
            else:
                for d in docs:
                    d.metadata["source"] = uploaded.name
                chunks = text_splitter(docs)
                create_embeddings(chunks)
                st.success(f"Ingested into {len(chunks)} chunks")
                st.session_state.chat_history = []
                st.session_state.messages = []

query = st.chat_input("Enter your query")
if query:
    with st.spinner("Thinking..."):
        answer, sources = generate_q(query, st.session_state.chat_history)
    st.session_state.messages.append((query, answer, sources))

# render the whole conversation as chat bubbles
for q, a, sources in st.session_state.messages:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("Sources"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}** — `{doc.metadata.get('source', '?')}` "
                            f"page {doc.metadata.get('page', '?')}")
                st.caption(doc.page_content[:500])