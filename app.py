"""
Advanced Streamlit App for Document Query Chatbot with Advanced RAG Techniques

This application demonstrates various advanced Retrieval-Augmented Generation (RAG) techniques,
including hybrid retrieval, query expansion, chain-of-thought query refinement, graph-based retrieval,
and document re-ranking. The code follows best practices in commenting and documentation to ensure 
clarity during presentations.
"""

# Standard library imports for file handling, debugging, and operating system utilities.
import streamlit as st
import tempfile          # Temporary file creation for document processing.
import os                # OS-related utilities, e.g., file paths and removal.
import time              # Time tracking if needed in performance analysis.
import traceback         # Detailed error traceback for debugging.
import shutil            # File and directory operations.

# Imports for document handling, vector search, and retrieval using LangChain and Chroma.
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader  # Loads various file formats.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits documents into chunks.
from langchain.retrievers import BM25Retriever, EnsembleRetriever  # Keyword and ensemble retrieval methods.
from langchain_community.retrievers import *  # Additional retriever functions.

from langchain.embeddings import CohereEmbeddings  # Embeddings for vector search using Cohere.
from langchain.llms import Cohere  # LLM for generation and query refinement from Cohere.

# Import the CrossEncoder model from sentence-transformers for document re-ranking.
from sentence_transformers import CrossEncoder

# NLTK is used for natural language processing tasks (like query expansion with WordNet).
import nltk
from nltk.corpus import wordnet

# Imports required for graph-based retrieval.
import spacy           # Natural Language Processing library for entity extraction.
import networkx as nx   # Library for graph creation and executing PageRank.

# Load spaCy's small English model for lightweight entity extraction.
nlp = spacy.load("en_core_web_sm")


##########################################
# Function: expand_query
##########################################
def expand_query(query):
    """
    Expand the user's query by appending synonyms using NLTK's WordNet.

    Steps:
      - Ensure WordNet and related corpora are downloaded.
      - For each word in the query, retrieve synonyms and add them to the query terms.
      - Return the expanded query as a single string.
      
    This expansion helps increase the chances of matching relevant document content by
    covering vocabulary variations.
    """
    try:
        # Check for the WordNet corpus; download if missing.
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        # Check for the omw corpus (for multilingual support) and download if missing.
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    
    # Split the input query into individual words.
    words = query.split()
    expanded_words = set(words)  # Use a set to avoid duplicates.
    
    # For every word, find synonyms via WordNet and add them (formatted correctly).
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    expanded_words.add(synonym)
    
    # Return the combined expanded query.
    return ' '.join(expanded_words)


##########################################
# Function: refine_query
##########################################
def refine_query(query, cohere_api_key):
    """
    Use an LLM (Cohere) to perform chain-of-thought reasoning on the user query and produce a refined version.
    
    The function:
      - Constructs a prompt asking the LLM to explain its reasoning and then output the refined query.
      - Sends the prompt to the Cohere LLM.
      - Parses the LLM's response for a line starting with "Refined Query:".
      - Returns the refined query (or falls back to the original query if not found).

    Parameters:
        query (str): The user's original query.
        cohere_api_key (str): API key for accessing Cohere's services.
    
    Returns:
        str: The refined query optimized for document retrieval.
    """
    prompt = (
        f"Below is a user's query: \"{query}\"\n\n"
        "Please perform chain-of-thought reasoning to understand the query and provide a refined version optimized for document retrieval. "
        "Explain your reasoning briefly, then on a separate line, output the refined query prefixed with 'Refined Query:'."
    )
    
    # Initialize the Cohere LLM with specified parameters.
    cohere_llm = Cohere(
        cohere_api_key=cohere_api_key,
        temperature=0.7,
        max_tokens=150
    )
    response = cohere_llm(prompt)
    
    # Attempt to extract the refined query from the LLM's response.
    refined_query = None
    for line in response.splitlines():
        if line.strip().startswith("Refined Query:"):
            refined_query = line.split("Refined Query:")[1].strip()
            break
    
    # Fallback: Return original query if no refinement was detected.
    if not refined_query:
        refined_query = query
    return refined_query


##########################################
# Function: load_and_preprocess_document
##########################################
def load_and_preprocess_document(uploaded_file):
    """
    Load an uploaded document and preprocess it for retrieval tasks.

    The function:
      - Saves the uploaded file temporarily.
      - Loads the document contents using UnstructuredFileLoader.
      - Cleans up the temporary file.
      - Splits the document into text chunks using a recursive character text splitter.
    
    Parameters:
        uploaded_file (file-like): The file uploaded by the user.
    
    Returns:
        list or None: A list of processed document chunks, or None in case of an error.
    """
    try:
        if uploaded_file is not None:
            # Determine file type and create a temporary file to store its contents.
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load the document from the temporary file.
            loader = UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()

            # Remove the temporary file now that it has been processed.
            os.remove(tmp_file_path)

            # Split the document into manageable chunks for retrieval.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            processed_docs = text_splitter.split_documents(documents)
            
            st.success(f"Document processed successfully. {len(processed_docs)} text chunks created.")
            return processed_docs
    except Exception as e:
        # Display an error message if document processing fails.
        st.error(f"Error processing the document: {e}")
        return None


##########################################
# Function: graph_based_retrieval
##########################################
def graph_based_retrieval(processed_docs, query):
    """
    Perform graph-based retrieval on document chunks using shared named entities.
    
    The function:
      - Extracts named entities from each document chunk using spaCy.
      - Constructs an undirected graph where each node represents a document chunk.
      - Adds edges between chunks if they share one or more entities, with edge weights indicating the count.
      - Extracts entities from the query and constructs a personalization vector.
      - Applies Personalized PageRank to rank document chunks in order of relevance.
    
    Parameters:
        processed_docs (list): List of document chunks.
        query (str): The user query used for personalizing the PageRank.
    
    Returns:
        list: A list of document chunks ranked by relevance.
    """
    # Map each document chunk to its set of named entities.
    chunk_entities = {}
    for idx, doc in enumerate(processed_docs):
        spacy_doc = nlp(doc.page_content)
        entities = {ent.text.lower() for ent in spacy_doc.ents}
        chunk_entities[idx] = entities

    # Build a graph where each node corresponds to a document chunk.
    G = nx.Graph()
    for idx in range(len(processed_docs)):
        G.add_node(idx, entities=chunk_entities[idx])
    
    # Connect nodes with an edge if their corresponding document chunks share common entities.
    for i in range(len(processed_docs)):
        for j in range(i + 1, len(processed_docs)):
            common_entities = chunk_entities[i].intersection(chunk_entities[j])
            if common_entities:
                # Edge weight is proportional to the number of shared entities.
                G.add_edge(i, j, weight=len(common_entities))
    
    # Extract named entities from the query text.
    query_doc = nlp(query)
    query_entities = {ent.text.lower() for ent in query_doc.ents}
    
    # Create a personalization dictionary for PageRank based on query entity overlap.
    personalization = {}
    for idx, entities in chunk_entities.items():
        score = len(query_entities.intersection(entities))
        personalization[idx] = score if score > 0 else 0.001  # Avoid zero values.
    
    # Run the Personalized PageRank algorithm on the constructed graph.
    pr = nx.pagerank(G, personalization=personalization, weight='weight')
    
    # Sort document chunks by their PageRank scores (highest score first).
    ranked_indices = sorted(pr, key=pr.get, reverse=True)
    ranked_docs = [processed_docs[i] for i in ranked_indices]
    return ranked_docs


##########################################
# Function: create_hybrid_retriever
##########################################
def create_hybrid_retriever(processed_docs, cohere_api_key):
    """
    Create a hybrid retriever that leverages both vector-based semantic search and BM25 keyword retrieval.
    
    This function:
      - Sets up a Chroma vector store using Cohere embeddings for semantic search.
      - Initializes a BM25 retriever for traditional keyword-based matching.
      - Combines these two approaches into an EnsembleRetriever with weighted contributions.
    
    Parameters:
        processed_docs (list): List of preprocessed document chunks.
        cohere_api_key (str): API key for obtaining Cohere embeddings.
    
    Returns:
        EnsembleRetriever or None: A combined retriever object, or None if an error occurs.
    """
    try:
        # Set up a directory to persist the Chroma vector database.
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize Cohere embeddings for document representation.
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0",
            client=None,
            user_agent="langchain"
        )
        
        # Build the vector store from processed documents using the embeddings.
        vectorstore = Chroma.from_documents(
            documents=processed_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="doc_collection"
        )
        
        # Set up the vector-based retriever with a fixed number of results.
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # Initialize the BM25 retriever for keyword-based search.
        bm25_retriever = BM25Retriever.from_documents(processed_docs)
        bm25_retriever.k = 3
        
        # Combine both retrievers into an ensemble with specified weights.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        # If an error occurs, display it along with a traceback for easier debugging.
        st.error(f"Error creating hybrid retriever: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None


##########################################
# Function: rerank_documents
##########################################
def rerank_documents(query, documents, cross_encoder):
    """
    Re-rank candidate document chunks based on their relevance to the query.
    
    The function:
      - Creates (query, document) pairs from the candidate documents.
      - Uses a pre-loaded CrossEncoder to predict a relevance score for each pair.
      - Sorts and returns the documents based on these scores in descending order.
    
    Parameters:
        query (str): The query text (typically the expanded query).
        documents (list): A list of candidate document chunks.
        cross_encoder (CrossEncoder): A pre-initialized CrossEncoder model.
    
    Returns:
        list: Sorted list of documents with the most relevant ones first.
    """
    try:
        # Prepare pairs of (query, document) for scoring.
        pairs = [(query, doc.page_content) for doc in documents]
        scores = cross_encoder.predict(pairs)
        # Combine documents and scores, then sort by score (high to low).
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, score in ranked_docs]
        return sorted_docs
    except Exception as e:
        # Log errors and fallback by returning the original document order.
        st.error(f"Error in re-ranking documents: {e}")
        return documents


##########################################
# Function: generate_response
##########################################
def generate_response(processed_docs, cohere_api_key, query_text):
    """
    Generate the final answer using advanced RAG techniques.
    
    This function integrates several steps:
      1. Chain-of-Thought Query Refinement:
           - Refines the original query using an LLM for an improved, retrieval-friendly query.
      2. Query Expansion:
           - Enhances the refined query by adding synonyms.
      3. Graph-Based Retrieval:
           - Ranks document chunks based on shared named entities using a graph-based approach.
      4. Hybrid Retrieval:
           - Retrieves candidate documents with an EnsembleRetriever combining semantic and keyword search.
      5. Re-Ranking:
           - Further refines the candidate list using a CrossEncoder to order them by contextual relevance.
      6. Answer Generation (Placeholder):
           - Uses the top-ranked documents as context for generating the final answer (actual generation logic can be integrated here).
    
    Parameters:
        processed_docs (list): Processed document chunks.
        cohere_api_key (str): API key for accessing Cohere services.
        query_text (str): The original user query.
    
    Returns:
        dict: A dictionary containing a final answer (placeholder) and source documents used.
    """
    try:
        # ----------------------------------------------------------------------------
        # Step 1: Query Refinement
        # Refine the query using chain-of-thought reasoning from the Cohere LLM.
        refined_query = refine_query(query_text, cohere_api_key)
        st.info(f"Refined Query: {refined_query}")
        
        # ----------------------------------------------------------------------------
        # Step 2: Query Expansion
        # Expand the refined query by adding synonyms (to cover vocabulary variations).
        expanded_query = expand_query(refined_query)
        st.info(f"Expanded Query: {expanded_query}")
        
        # ----------------------------------------------------------------------------
        # Step 3: Graph-Based Retrieval
        # Compute rankings using graph-based retrieval based on shared named entities.
        graph_ranked_docs = graph_based_retrieval(processed_docs, expanded_query)
        
        # ----------------------------------------------------------------------------
        # Step 4: Hybrid Retrieval
        # Retrieve additional candidate documents using the EnsembleRetriever.
        hybrid_retriever = create_hybrid_retriever(processed_docs, cohere_api_key)
        if hybrid_retriever is None:
            return None
        candidate_docs = hybrid_retriever.get_relevant_documents(expanded_query)
        
        # ----------------------------------------------------------------------------
        # Step 5: Combine and Deduplicate Candidate Documents
        # Merge the documents obtained from graph-based and hybrid retrieval, ensuring uniqueness.
        combined_docs = list({doc.page_content: doc for doc in (graph_ranked_docs + candidate_docs)}.values())
        
        # ----------------------------------------------------------------------------
        # Step 6: Re-Ranking
        # Apply a CrossEncoder model to re-rank the combined candidate documents.
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranked_docs = rerank_documents(expanded_query, combined_docs, cross_encoder)
        top_docs = ranked_docs[:5]  # Select the top 5 most relevant documents.
        
        # ----------------------------------------------------------------------------
        # Step 7: Answer Generation (Placeholder)
        # Here, you would typically generate an answer using a language model (e.g., via a RetrievalQA chain)
        # by combining the content of the top-ranked documents. For now, we return a placeholder message.
        return {"result": "Answer generation logic goes here", "source_documents": top_docs}
    
    except Exception as e:
        # Log any exception to the UI.
        st.error(f"Error generating response: {e}")
        return None


##########################################
# Function: cleanup_chroma_db
##########################################
def cleanup_chroma_db():
    """
    Clean up the ChromaDB directory by removing the persisted Chroma vector store.
    
    This is useful for resetting the database between sessions or cleaning up resources.
    """
    try:
        chroma_db_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_db_dir):
            shutil.rmtree(chroma_db_dir)
            st.success("ChromaDB directory cleaned up successfully.")
    except Exception as e:
        st.error(f"Error cleaning up ChromaDB directory: {e}")


##############################
# Streamlit App UI
##############################
st.set_page_config(page_title='🦜🔗 Advanced Document Query Chatbot')
st.title('🦜🔗 Advanced Document Query Chatbot')

st.write("## Instructions")
st.write(
    "Upload a document (txt, pdf, docx, or doc) and ask a question about its content. "
    "This version now includes advanced techniques: agentic query refinement (chain-of-thought) "
    "and graph-based retrieval, along with candidate re-ranking and query expansion."
)

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
        with st.spinner('Processing document, refining query, retrieving documents, and generating answer...'):
            processed_docs = load_and_preprocess_document(uploaded_file)
            if processed_docs:
                result = generate_response(processed_docs, cohere_api_key, query_text)

if result:
    st.write("## Answer")
    st.write(result["result"])
    
    st.write("### Source Document Chunks Used (Re-ranked):")
    for i, doc in enumerate(result.get("source_documents", []), start=1):
        st.write(f"**Chunk {i}:**")
        st.markdown(doc.page_content)

with st.expander("Show Evaluation Metrics (Basic)"):
    st.write("Here, you can input test questions and expected answers to evaluate the chatbot's performance.")
    test_question = st.text_input("Test Question", placeholder="Enter a test question here...")
    expected_answer = st.text_area("Expected Answer", placeholder="Enter the expected answer here...")
    evaluate = st.button("Evaluate Test Query")
    
    if evaluate and test_question and expected_answer and result:
        if expected_answer.lower() in result["result"].lower():
            st.success("The answer meets the expected criteria.")
        else:
            st.warning("The answer does not match the expected answer well. Further tuning may be required.")

st.write("----")
st.write("**Note:** This prototype demonstrates advanced techniques including agentic query refinement, "
         "graph-based retrieval, candidate re-ranking, query expansion, and basic evaluation integration. "
         "Further enhancements (e.g., multi-modal parsing) can be added in future iterations.")

if __name__ == "__main__":
    try:
        pass
    finally:
        cleanup_chroma_db()
