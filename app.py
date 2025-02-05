import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Optional: For OCR image handling, you may import pytesseract and pdf2image if needed
# import pytesseract
# from pdf2image import convert_from_path

def load_and_preprocess_document(uploaded_file):
    """
    Load and preprocess the uploaded file using UnstructuredFileLoader and
    RecursiveCharacterTextSplitter for better chunking.
    Added error handling and user feedback.
    """
    try:
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file to allow file-based loading.
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Use UnstructuredFileLoader which infers file type from the extension.
            loader = UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()

            # Remove the temporary file after loading.
            os.remove(tmp_file_path)

            # Chunk the document with a recursive text splitter.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            processed_docs = text_splitter.split_documents(documents)

            # Provide user feedback about successful processing
            st.success(f"Document processed successfully. {len(processed_docs)} text chunks created.")
            return processed_docs
    except Exception as e:
        st.error(f"Error processing the document: {e}")
        return None

def create_hybrid_retriever(processed_docs, google_api_key):
    """
    Create a hybrid retriever that uses both vector search (Chroma) and BM25 retrieval.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vectorstore = Chroma.from_documents(processed_docs, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        bm25_retriever = BM25Retriever.from_documents(processed_docs)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        st.error(f"Error creating hybrid retriever: {e}")
        return None

def generate_response(processed_docs, google_api_key, query_text):
    """
    Generate the answer using RetrievalQA with the hybrid retriever.
    Modified to return both the answer and the source documents used.
    """
    try:
        retriever = create_hybrid_retriever(processed_docs, google_api_key)
        if retriever is None:
            return None
        # Set return_source_documents=True to get source context
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({"query": query_text})
        return result  # result contains "result" and "source_documents"
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title='🦜🔗 Advanced Ask the Doc App v3 - Gemini')
st.title('🦜🔗 Advanced Ask the Doc App v3 - Gemini')

st.write("## Instructions")
st.write(
    "Upload a document (txt, pdf, docx, or doc) and ask a question about its content. "
    "This version uses **Google Gemini Pro** for question answering."
    " It includes advanced error handling, feedback on document processing, and displays the source context "
    "used to generate the answer."
)

# File upload - Keep file upload OUTSIDE the form for now
uploaded_file = st.file_uploader('Upload a document', type=['txt', 'pdf', 'docx', 'doc'])

result = None

with st.form('query_form', clear_on_submit=True): # Query and API Key are now IN the form
    query_text = st.text_input('Enter your question:', placeholder='Ask something about the document.', disabled=(uploaded_file is None)) # Moved inside form
    google_api_key = st.text_input(
        'Google API Key',
        type='password',
        help="Enter your Google API key (Get it from Google AI Studio).",
        disabled=False
    )
    st.write(f"uploaded_file: {uploaded_file is not None}") # Debugging - Check if file is uploaded
    st.write(f"query_text: {bool(query_text)}")       # Debugging - Check if query is entered
    st.write(f"google_api_key: {bool(google_api_key)}")   # Debugging - Check if API key is entered
    st.write(f"Combined condition: {(uploaded_file and query_text and google_api_key)}") # Debugging - Combined condition (including uploaded_file now, even though file upload is outside form)
    st.write(f"disabled: {not google_api_key}") # Debugging - Simplified disabled condition - ONLY depends on API key now for testing
    submitted = st.form_submit_button("Submit Query", disabled=(not google_api_key)) # Simplified disabled condition - ONLY depends on API key for testing
    if submitted:
        with st.spinner('Processing document and generating answer...'):
            processed_docs = load_and_preprocess_document(uploaded_file)
            if processed_docs:
                result = generate_response(processed_docs, google_api_key, query_text)

if result:
    st.write("## Answer")
    # Display answer
    answer = result.get("result", "No answer returned")
    st.info(answer)

    # Display source context
    source_docs = result.get("source_documents", [])
    if source_docs:
        st.write("### Source Document Chunks Used:")
        for i, doc in enumerate(source_docs, start=1):
            st.write(f"**Chunk {i}:**")
            st.markdown(doc.page_content)  # Assuming each document has a 'page_content' attribute

# --- Evaluation Section (Basic Evaluation Integration) ---
with st.expander("Show Evaluation Metrics (Basic)"):
    st.write("Here, you can input test questions and expected answers to evaluate the chatbot's performance.")
    test_question = st.text_input("Test Question", placeholder="Enter a test question here...")
    expected_answer = st.text_area("Expected Answer", placeholder="Enter the expected answer here...")
    evaluate = st.button("Evaluate Test Query")

    if evaluate and test_question and expected_answer and result:
        # For simplicity, we compare if the answer contains expected answer keywords
        if expected_answer.lower() in answer.lower():
            st.success("The answer meets the expected criteria.")
        else:
            st.warning("The answer does not match the expected answer well. Further tuning may be required.")

st.write("----")
st.write("**Note:** This is a prototype demonstrating advanced features using **Google Gemini Pro**, including improved error handling, source context display, and basic evaluation integration. Further enhancements (such as image OCR and advanced table parsing) can be added in future iterations.")