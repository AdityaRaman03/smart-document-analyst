import tempfile
import os
from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_documents(pdf_bytes):
    '''Document Loaders supported by Langchain can only read from the disk.
    A stream of data is extracted and stored in a temporary location so the Loaders can access it.'''

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush()
        tmp_path = temp_pdf.name

    try:
        loader = PyMuPDF4LLMLoader(tmp_path, extract_images=False)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    if len(docs) == 0:  # no extractable text (e.g. a scanned/image-only PDF)
        print("Please input valid text-based pdf")
        return []

    return docs


def text_splitter(docs):
    # split into 1000-char chunks with 150-char overlap; split_documents keeps page metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return chunks

def create_embeddings(chunks, persist_dir="./chroma_db"):
    # embed each chunk with Gemini and persist the vectors to Chroma on disk
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vectorstore

# Standalone test only (run: python ingestion_pipeline.py) — not used by the app.
# if __name__ == "__main__":
#     filepath = "/home/aditya/Downloads/rag_test_doc.pdf"
#
#     with open(filepath, "rb") as f:          # read the file as bytes
#         pdf_bytes = f.read()
#
#     docs = load_documents(pdf_bytes)          # pass bytes, not the path
#     chunks = text_splitter(docs)
#     vectorstore = create_embeddings(chunks)
#
#     # sanity check that retrieval works
#     results = vectorstore.similarity_search("What is chunk overlap?", k=2)
#     for r in results:
#         print(r.page_content[:200], "\n---")