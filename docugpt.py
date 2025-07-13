import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Local Embedding Wrapper
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]

# Streamlit UI
st.set_page_config(page_title="DocuGPT", layout="centered")
st.header("📄 DocuGPT")

with st.sidebar:
    st.title("My Notes")
    uploaded_file=st.file_uploader("Upload Notes PDF and start asking questions",type="pdf")

if uploaded_file is not None:
    # Step 1: Extract text
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    # Step 2: Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    if not chunks:
        st.error("⚠️ No text could be extracted from the PDF.")
        st.stop()

    # Step 3: Create vector store
    try:
        embeddings = LocalEmbeddings()
        vector_store = Chroma.from_texts(chunks, embeddings)
        st.success("✅ Document uploaded successfully")
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        st.stop()

    # Step 4: Ask user input
    question = st.text_input("Ask a question about the PDF:")

    if question:
        try:
            hf_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                tokenizer="google/flan-t5-base",
                max_length=512,
                temperature=0.5,
            )

            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            docs = vector_store.similarity_search(question)
            response = chain.run(input_documents=docs, question=question)

            st.markdown("**Answer:**")
            st.info(response)

        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
