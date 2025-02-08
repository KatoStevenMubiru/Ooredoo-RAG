import streamlit as st
import tempfile
import os
import time
import traceback
import shutil

# Import necessary modules for document processing and retrieval
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import *
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere

# Import CrossEncoder from sentence-transformers for re-ranking
from sentence_transformers import CrossEncoder

# Import NLTK for query expansion using WordNet
import nltk
from nltk.corpus import wordnet

def expand_query(query):
    """
    Expand the user's query by adding synonyms using NLTK's WordNet.
    
    Steps:
    - Ensure the WordNet corpora are downloaded.
    - For each word in the query, retrieve synonyms.
    - Combine the original words with their synonyms and return the expanded query.
    """
    # Ensure WordNet data is available
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    
    words = query.split()
    expanded_words = set(words)  # Use a set to avoid duplicates
    
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    expanded_words.add(synonym)
    
    # Return the expanded query as a string
    return ' '.join(expanded_words)

def load_and_preprocess_document(uploaded_file):
    """
    Load and preprocess the uploaded file using UnstructuredFileLoader and 
    RecursiveCharacterTextSplitter for better chunking.
    
    Steps:
    - Save the uploaded file temporarily.
    - Use UnstructuredFileLoader to load the document.
    - Remove the temporary file.
    - Split the document into manageable text chunks.
    - Provide user feedback on success.
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

            # Chunk the document using a recursive text splitter.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            processed_docs = text_splitter.split_documents(documents)
            
            st.success(f"Document processed successfully. {len(processed_docs)} text chunks created.")
            return processed_docs
    except Exception as e:
        st.error(f"Error processing the document: {e}")
        return None

def create_hybrid_retriever(processed_docs, cohere_api_key):
    """
    Create a hybrid retriever that uses both vector search (Chroma) and BM25 retrieval.
    
    Steps:
    - Create a persistent directory for ChromaDB.
    - Initialize Cohere embeddings.
    - Build the vectorstore (Chroma) and its retriever.
    - Initialize a BM25 retriever from the processed documents.
    - Combine both retrievers using EnsembleRetriever with defined weights.
    """
    try:
        # Create a persistent directory for ChromaDB
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embeddings using CohereEmbeddings
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
        st.error(f"Error creating hybrid retriever: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None

def rerank_documents(query, documents, cross_encoder):
    """
    Rerank the list of documents based on relevance to the query using a CrossEncoder.
    
    Steps:
    - Create (query, document) pairs for all candidate documents.
    - Use the cross encoder to predict a relevance score for each pair.
    - Sort the documents by their score in descending order.
    - Return the sorted list of documents.
    """
    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = cross_encoder.predict(pairs)
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, score in ranked_docs]
        return sorted_docs
    except Exception as e:
        st.error(f"Error in re-ranking documents: {e}")
        return documents

def generate_response(processed_docs, cohere_api_key, query_text):
    """
    Generate the answer using a two-step retrieval process with query expansion:
      1. Expand the user's query to include synonyms.
      2. Retrieve candidate documents using the expanded query.
      3. Re-rank the retrieved documents using a CrossEncoder.
      4. Generate the final answer using the top-ranked documents.
      
    The answer is streamed with a typing effect for a dynamic UI.
    """
    try:
        # Step 1: Expand the query using synonyms
        expanded_query = expand_query(query_text)
        st.info(f"Expanded Query: {expanded_query}")
        
        # Create the hybrid retriever
        retriever = create_hybrid_retriever(processed_docs, cohere_api_key)
        if retriever is None:
            return None
            
        # Step 2: Retrieve candidate documents using the expanded query
        candidate_docs = retriever.get_relevant_documents(expanded_query)
        
        # Step 3: Re-rank the candidate documents using a CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranked_docs = rerank_documents(expanded_query, candidate_docs, cross_encoder)
        top_docs = ranked_docs[:5]
        
        # Prepare the context text from the top-ranked documents
        docs_text = "\n\n".join([doc.page_content for doc in top_docs])
        # Construct the prompt using the original query (for clarity) along with the document context
        prompt = f"Based on the following documents, answer the question:\n\n{docs_text}\n\nQuestion: {query_text}"
        
        # Initialize the Cohere LLM for generating the answer
        cohere_llm = Cohere(
            cohere_api_key=cohere_api_key,
            temperature=0.7,
            max_tokens=512
        )
        
        # Generate the response using the LLM
        result_text = cohere_llm(prompt)
        
        # Stream the answer with a typing effect
        response_placeholder = st.empty()
        displayed_text = ""
        for char in result_text:
            displayed_text += char
            response_placeholder.info(displayed_text + "▌")
            time.sleep(0.01)  # Adjust the typing speed as needed
            
        # Final update without the cursor effect
        response_placeholder.info(displayed_text)
        
        return {"result": result_text, "source_documents": top_docs}
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None

def cleanup_chroma_db():
    """
    Clean up the ChromaDB directory when needed.
    """
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    if os.path.exists(persist_directory):
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
    "This version includes advanced error handling, candidate re-ranking, query expansion, "
    "and displays the source context used to generate the answer."
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
        with st.spinner('Processing document, expanding query, retrieving and re-ranking candidate chunks, and generating answer...'):
            processed_docs = load_and_preprocess_document(uploaded_file)
            if processed_docs:
                result = generate_response(processed_docs, cohere_api_key, query_text)

if result:
    st.write("## Answer")
    # The answer is already displayed via the streaming effect above.
    
    # Display the re-ranked source document chunks used
    source_docs = result.get("source_documents", [])
    if source_docs:
        st.write("### Source Document Chunks Used (Re-ranked):")
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
        # A simple keyword matching to compare the answer with the expected answer.
        if expected_answer.lower() in result["result"].lower():
            st.success("The answer meets the expected criteria.")
        else:
            st.warning("The answer does not match the expected answer well. Further tuning may be required.")

st.write("----")
st.write("**Note:** This prototype demonstrates advanced features including improved error handling, candidate re-ranking, "
         "query expansion, source context display, and basic evaluation integration. Additional enhancements (like OCR or advanced table parsing) can be added in future iterations.")

# Clean up the ChromaDB persistence on exit
if __name__ == "__main__":
    try:
        pass
    finally:
        cleanup_chroma_db()
