import streamlit as st
import tempfile
import os
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import *
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
import time

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

def create_hybrid_retriever(processed_docs, cohere_api_key):
    """
    Create a hybrid retriever that uses both vector search (Chroma) and BM25 retrieval.
    """
    try:
        # Create a persistent directory for ChromaDB
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        # Updated embeddings initialization with additional required parameters
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0",
            client=None,
            user_agent="langchain"
        )
        
        # Initialize Chroma with persistent directory
        vectorstore = Chroma.from_documents(
            documents=processed_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="doc_collection"
        )
        
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        bm25_retriever = BM25Retriever.from_documents(processed_docs)
        bm25_retriever.k = 3
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        import traceback
        st.error(f"Error creating hybrid retriever: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None

def generate_response(processed_docs, cohere_api_key, query_text):
    """
    Generate the answer using RetrievalQA with the hybrid retriever.
    Modified to stream the response.
    """
    try:
        retriever = create_hybrid_retriever(processed_docs, cohere_api_key)
        if retriever is None:
            return None
            
        # Updated Cohere LLM initialization without streaming parameter
        cohere_llm = Cohere(
            cohere_api_key=cohere_api_key,
            temperature=0.7,
            max_tokens=512
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=cohere_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Create a placeholder for displaying the response
        response_placeholder = st.empty()
        
        # Get the response
        result = qa({"query": query_text})
        
        # Display the response with a typing effect
        response_text = result["result"]
        displayed_text = ""
        for i in range(len(response_text)):
            displayed_text += response_text[i]
            response_placeholder.info(displayed_text + "▌")
            time.sleep(0.01)  # Adjust the speed of typing
            
        # Final update without the cursor
        response_placeholder.info(displayed_text)
        
        return result
        
    except Exception as e:
        import traceback
        st.error(f"Error generating response: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None

# Add cleanup function to manage ChromaDB persistence
def cleanup_chroma_db():
    """
    Clean up the ChromaDB directory when needed
    """
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    if os.path.exists(persist_directory):
        import shutil
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            st.warning(f"Error cleaning up ChromaDB: {e}")

# --- Streamlit App UI ---
st.set_page_config(page_title='🦜🔗 Advanced Document Query Chatbot')
st.title('🦜🔗 Advanced Document Query Chatbot')

st.write("## Instructions")
st.write(
    "Upload a document (txt, pdf, docx, or doc) and ask a question about its content. "
    "This version includes advanced error handling, feedback on document processing, and displays the source context "
    "used to generate the answer."
)

# File upload and query input
uploaded_file = st.file_uploader('Upload a document', type=['txt', 'pdf', 'docx', 'doc'])
query_text = st.text_input('Enter your question:', placeholder='Ask something about the document.', disabled=(uploaded_file is None))

result = None

with st.form('query_form', clear_on_submit=True):
    cohere_api_key = st.text_input(
        'Cohere API Key',
        type='password',
        help="Enter your Cohere API key (must start with x-).",
        disabled=False
    )
    submitted = st.form_submit_button("Submit Query")
    if submitted:
        with st.spinner('Processing document and generating answer...'):
            processed_docs = load_and_preprocess_document(uploaded_file)
            if processed_docs:
                result = generate_response(processed_docs, cohere_api_key, query_text)

if result:
    st.write("## Answer")
    # Answer is already displayed through streaming
    
    # Display source context
    source_docs = result.get("source_documents", [])
    if source_docs:
        st.write("### Source Document Chunks Used:")
        for i, doc in enumerate(source_docs, start=1):
            st.write(f"**Chunk {i}:**")
            st.markdown(doc.page_content)

# --- Evaluation Section (Basic Evaluation Integration) ---
with st.expander("Show Evaluation Metrics (Basic)"):
    st.write("Here, you can input test questions and expected answers to evaluate the chatbot's performance.")
    test_question = st.text_input("Test Question", placeholder="Enter a test question here...")
    expected_answer = st.text_area("Expected Answer", placeholder="Enter the expected answer here...")
    evaluate = st.button("Evaluate Test Query")
    
    if evaluate and test_question and expected_answer and result:
        # For simplicity, we compare if the answer contains expected answer keywords
        # In a real-world scenario, you would compute more robust metrics
        if expected_answer.lower() in result["result"].lower():
            st.success("The answer meets the expected criteria.")
        else:
            st.warning("The answer does not match the expected answer well. Further tuning may be required.")

st.write("----")
st.write("**Note:** This is a prototype demonstrating advanced features, including improved error handling, source context display, and basic evaluation integration. Further enhancements (such as image OCR and advanced table parsing) can be added in future iterations.")

# Remove the main() function call and use this instead
if __name__ == "__main__":
    try:
        # Your Streamlit app code is already running, no need for a main() function
        pass
    finally:
        cleanup_chroma_db()