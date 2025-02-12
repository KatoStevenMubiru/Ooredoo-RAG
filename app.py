"""
Advanced Streamlit App for Document Query Chatbot with Advanced RAG Techniques

This application demonstrates various advanced Retrieval-Augmented Generation (RAG) techniques,
including hybrid retrieval, query expansion, chain-of-thought query refinement, graph-based retrieval,
advanced document content handling, multi-step reasoning and query decomposition, and evaluation.
The code is extensively commented for clarity.
"""

# --- Streamlit Configuration ---
import streamlit as st

# Set page configuration as the very first Streamlit command.
st.set_page_config(page_title="🦜🔗 Advanced Document Query Chatbot")

# --- Standard Library Imports ---
import tempfile  # For temporary file handling.
import os  # For file path operations.
import time  # For delays, performance measurement, and response time tracking.
import traceback  # For detailed error tracebacks.
import shutil  # For file/directory cleanup.
import re  # For regex-based text cleaning.
import string  # For punctuation definitions.

# --- Imports for Document Processing and Retrieval ---
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import (
    UnstructuredFileLoader,
)  # (Deprecated, but kept for compatibility)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import *  # Additional retriever functions.
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere

# --- Import for Re-Ranking ---
from sentence_transformers import CrossEncoder

# --- NLTK for Query Expansion ---
import nltk
from nltk.corpus import wordnet

# --- Imports for Graph-Based Retrieval and Entity Extraction ---
import spacy
import networkx as nx

nlp = spacy.load(
    "en_core_web_sm"
)  # Load spaCy's small English model for entity extraction.

# --- Imports for Advanced Document Content Handling ---
import cv2
import easyocr

try:
    from camelot.io import read_pdf  # For table extraction from PDFs.
except ImportError:
    st.error("Camelot not installed. Please install with: pip install 'camelot-py[cv]'")


# --- Helper Function: Text Cleaning ---
def clean_text(text):
    """
    Clean the input text by lowercasing, removing punctuation, and normalizing whitespace.

    Parameters:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Convert text to lowercase.
    text = text.lower()
    # Remove punctuation using regex (ensures that punctuation doesn't affect token matching).
    text = re.sub(r"[" + re.escape(string.punctuation) + "]", "", text)
    # Normalize whitespace: remove extra spaces, newlines, and tabs.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Advanced Document Content Handler Class ---
class AdvancedContentHandler:
    """
    Processes documents to extract advanced content such as tables and image text.
    Uses Camelot for table extraction (from PDFs) and EasyOCR for OCR on images.
    """

    def __init__(self):
        self.table_data = []  # List to store extracted tables.
        self.image_data = []  # List to store extracted text from images.
        try:
            # Initialize EasyOCR reader (set gpu=True if a GPU is available).
            self.reader = easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            st.error(
                f"Error initializing EasyOCR: {e}. Image text extraction will be disabled."
            )
            self.reader = None

    def process_document(self, file_path):
        """
        Process the document file to extract tables (if it's a PDF) and image text.

        Parameters:
            file_path (str): Path to the document file.
        """
        # If the file is a PDF, attempt to extract tables using Camelot.
        if file_path.endswith(".pdf"):
            try:
                tables = read_pdf(file_path, flavor="lattice")
                self.table_data = [table.df for table in tables]
            except Exception as e:
                st.warning(f"Error extracting tables: {e}")

        # Attempt to read the file as an image and extract text using EasyOCR.
        img = cv2.imread(file_path)
        if img is not None and self.reader is not None:
            self.extract_image_text(img)

    def extract_image_text(self, image):
        """
        Extract text from the image using EasyOCR.

        Parameters:
            image (numpy array): The image data.
        """
        try:
            result = self.reader.readtext(image, detail=0)
            text = " ".join(result)
            self.image_data.append({"type": "image_text", "content": text})
        except Exception as e:
            st.warning(f"Error extracting text from image: {e}")


# --- Comprehensive Evaluation Framework ---
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import numpy as np


class RAGEvaluator:
    """
    Evaluates the performance of the RAG system.
    Computes metrics including retrieval precision/recall, semantic similarity, and response time.
    """

    def __init__(self):
        self.metrics_history = []
        # Use a SentenceTransformer model to compute embeddings for answer quality evaluation.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate_iteration(self, test_cases, responses ):
        """
        Evaluate a single iteration's performance and generate a report.

        Parameters:
            test_cases (list): List of test cases with 'relevant_documents' and 'expected_answer'.
            responses (list): List of system responses.
            elapsed_time (float): Time taken to generate the answer (in seconds).

        Returns:
            str: Formatted evaluation report.
        """
        metrics = {
            "retrieval_metrics": self.evaluate_retrieval(test_cases, responses),
            "answer_quality": self.evaluate_answer_quality(test_cases, responses),
            "response_time": {"average_time": 0.0},
        }
        self.metrics_history.append({"timestamp": time.time(), "metrics": metrics})
        return self.generate_evaluation_report(metrics)

    def evaluate_retrieval(self, test_cases, responses):
        precision_scores = []
        recall_scores = []
        for test_case, response in zip(test_cases, responses):
            relevant_docs = set(test_case["relevant_documents"])
            retrieved_docs = set(response["source_documents"])
            precision = len(relevant_docs & retrieved_docs) / (len(retrieved_docs) or 1)
            recall = len(relevant_docs & retrieved_docs) / (len(relevant_docs) or 1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        return {
            "average_precision": np.mean(precision_scores),
            "average_recall": np.mean(recall_scores),
        }

    def evaluate_answer_quality(self, test_cases, responses):
        similarities = []
        for test_case, response in zip(test_cases, responses):
            expected_embedding = self.model.encode(test_case["expected_answer"])
            actual_embedding = self.model.encode(response["answer"])
            similarity = np.dot(expected_embedding, actual_embedding) / (
                np.linalg.norm(expected_embedding) * np.linalg.norm(actual_embedding)
                or 1
            )
            similarities.append(similarity)
        return {"semantic_similarity": np.mean(similarities)}

    def measure_performance(self, test_cases):
        # This function is replaced by passing elapsed_time directly.
        return {"average_time": 0.0}

    def generate_evaluation_report(self, metrics):
        report = (
            f"Retrieval - Precision: {metrics['retrieval_metrics']['average_precision']:.2f}, "
            f"Recall: {metrics['retrieval_metrics']['average_recall']:.2f}\n"
            f"Answer Quality - Semantic Similarity: {metrics['answer_quality']['semantic_similarity']:.2f}\n"
            f"Response Time - Average: {metrics['response_time']['average_time']:.2f} sec"
        )
        return report


# --- Multi-Step Reasoning and Query Decomposition ---
class ComplexQueryHandler:
    """
    Decomposes a complex query into sub-queries using the LLM, builds a reasoning chain (DAG),
    and synthesizes a final answer from intermediate results.
    """

    def __init__(self, llm):
        self.llm = llm
        self.reasoning_steps = []

    def process_complex_query(self, query: str) -> dict:
        sub_queries = self.decompose_query(query)
        reasoning_chain = self.build_reasoning_chain(sub_queries)
        final_result = self.execute_reasoning_chain(reasoning_chain)
        return self.verify_and_synthesize(final_result)

    def decompose_query(self, query: str) -> list:
        prompt = f"Decompose this query into simpler sub-questions:\nQuery: {query}\nBreak it down into sequential steps."
        response = self.llm(prompt)
        return self.parse_sub_queries(response)

    def parse_sub_queries(self, response: str) -> list:
        # Split the response into non-empty lines.
        return [line.strip() for line in response.splitlines() if line.strip()]

    def build_reasoning_chain(self, sub_queries: list) -> nx.DiGraph:
        graph = nx.DiGraph()
        for i, q in enumerate(sub_queries):
            graph.add_node(i, query=q)
            if i > 0:
                graph.add_edge(i - 1, i)
        return graph

    def execute_reasoning_chain(self, chain: nx.DiGraph) -> dict:
        results = {}
        for node in nx.topological_sort(chain):
            query = chain.nodes[node]["query"]
            context = self.get_context_from_previous(node, chain, results)
            result = self.execute_single_query(query, context)
            results[node] = result
            self.reasoning_steps.append(
                {"step": node, "query": query, "result": result}
            )
        return results

    def get_context_from_previous(self, node, chain, results):
        # Concatenate results from all predecessor nodes.
        return " ".join(str(results.get(prev, "")) for prev in chain.predecessors(node))

    def execute_single_query(self, query, context):
        prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
        return self.llm(prompt)

    def verify_and_synthesize(self, results: dict) -> dict:
        if results:
            final_node = max(results.keys())
            return {"final_answer": results[final_node], "steps": self.reasoning_steps}
        return {"final_answer": "", "steps": []}


# --- Existing Functions: expand_query, refine_query, load_and_preprocess_document, graph_based_retrieval, create_hybrid_retriever, rerank_documents ---


def expand_query(query):
    """
    Expand the user's query by appending synonyms using NLTK's WordNet.
    This increases the chances of matching document content with different vocabulary.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")
    words = query.split()
    expanded_words = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    expanded_words.add(synonym)
    return " ".join(expanded_words)


def refine_query(query, cohere_api_key):
    """
    Use a Cohere LLM with chain-of-thought prompting to refine the user's query.
    The prompt instructs the LLM to explain its reasoning and output a refined query.
    """
    prompt = (
        f'Below is a user\'s query: "{query}"\n\n'
        "Please perform chain-of-thought reasoning to understand the query and provide a refined version optimized for document retrieval. "
        "Explain your reasoning briefly, then on a separate line, output the refined query prefixed with 'Refined Query:'."
    )
    cohere_llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.7, max_tokens=150)
    response = cohere_llm(prompt)
    refined_query = None
    for line in response.splitlines():
        if line.strip().startswith("Refined Query:"):
            refined_query = line.split("Refined Query:")[1].strip()
            break
    if not refined_query:
        refined_query = query
    return refined_query


def load_and_preprocess_document(uploaded_file):
    """
    Load an uploaded document, clean its text, and preprocess it for retrieval tasks.

    Steps:
      - Save the uploaded file temporarily.
      - Load the document contents using UnstructuredFileLoader.
      - Remove the temporary file.
      - Combine the document text and apply cleaning (lowercasing, punctuation removal, whitespace normalization).
      - Split the cleaned text into chunks using RecursiveCharacterTextSplitter.

    Returns:
        A list of processed document chunks.
    """
    try:
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            # Load document content using UnstructuredFileLoader.
            loader = UnstructuredFileLoader(tmp_file_path)
            documents = loader.load()
            os.remove(tmp_file_path)
            # Combine all document chunks into one text.
            combined_text = " ".join(doc.page_content for doc in documents)
            # Clean the text using our clean_text helper.
            cleaned_text = clean_text(combined_text)
            # Create a new Document with the cleaned text.
            from langchain.docstore.document import Document

            cleaned_document = Document(page_content=cleaned_text)
            cleaned_documents = [cleaned_document]
            # Split the cleaned document into smaller chunks.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            processed_docs = text_splitter.split_documents(cleaned_documents)
            st.success(
                f"Document processed successfully. {len(processed_docs)} text chunks created."
            )
            return processed_docs
    except Exception as e:
        st.error(f"Error processing the document: {e}")
        return None


def graph_based_retrieval(processed_docs, query):
    """
    Perform graph-based retrieval on document chunks using shared named entities.

    Steps:
      - Extract named entities from each chunk using spaCy.
      - Build an undirected graph where each node represents a chunk.
      - Add edges between nodes if they share common entities.
      - Use Personalized PageRank (with query entity overlap) to rank the chunks.

    Returns:
        A list of document chunks ranked by relevance.
    """
    chunk_entities = {}
    for idx, doc in enumerate(processed_docs):
        spacy_doc = nlp(doc.page_content)
        entities = {ent.text.lower() for ent in spacy_doc.ents}
        chunk_entities[idx] = entities
    G = nx.Graph()
    for idx in range(len(processed_docs)):
        G.add_node(idx, entities=chunk_entities[idx])
    for i in range(len(processed_docs)):
        for j in range(i + 1, len(processed_docs)):
            common_entities = chunk_entities[i].intersection(chunk_entities[j])
            if common_entities:
                G.add_edge(i, j, weight=len(common_entities))
    query_doc = nlp(query)
    query_entities = {ent.text.lower() for ent in query_doc.ents}
    personalization = {}
    for idx, entities in chunk_entities.items():
        score = len(query_entities.intersection(entities))
        personalization[idx] = score if score > 0 else 0.001
    pr = nx.pagerank(G, personalization=personalization, weight="weight")
    ranked_indices = sorted(pr, key=pr.get, reverse=True)
    ranked_docs = [processed_docs[i] for i in ranked_indices]
    return ranked_docs


def create_hybrid_retriever(processed_docs, cohere_api_key):
    """
    Create a hybrid retriever combining vector-based semantic search and BM25 keyword retrieval.

    Steps:
      - Set up a Chroma vector store using CohereEmbeddings for semantic search.
      - Initialize a BM25 retriever for keyword-based search.
      - Combine both retrievers using an EnsembleRetriever.

    Returns:
        An EnsembleRetriever object.
    """
    try:
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0",
            client=None,
            user_agent="langchain",
        )
        vectorstore = Chroma.from_documents(
            documents=processed_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="doc_collection",
        )
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        bm25_retriever = BM25Retriever.from_documents(processed_docs)
        bm25_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.3]
        )
        return ensemble_retriever
    except Exception as e:
        st.error(
            f"Error creating hybrid retriever: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        )
        return None


def rerank_documents(query, documents, cross_encoder):
    """
    Re-rank candidate document chunks based on their relevance to the query using a CrossEncoder.

    Returns:
        A sorted list of documents with the most relevant ones first.
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
    Generate the final answer using advanced RAG techniques.

    Steps:
      1. Chain-of-Thought Query Refinement: Refine the query using an LLM.
      2. Query Expansion: Enhance the query with synonyms.
      3. Graph-Based Retrieval: Rank document chunks via shared entities.
      4. Hybrid Retrieval: Retrieve candidate chunks via vector and BM25 methods.
      5. Combine and Deduplicate Candidate Documents.
      6. Re-Rank the candidates using a CrossEncoder.
      7. Final Answer Generation: Use the top-ranked chunks as context for generating the answer.

    Returns:
        A dictionary containing the final answer and the source document chunks.
    """
    try:
        # Step 1: Chain-of-Thought Query Refinement
        refined_query = refine_query(query_text, cohere_api_key)
        st.info(f"Refined Query: {refined_query}")

        # Step 2: Query Expansion
        expanded_query = expand_query(refined_query)
        st.info(f"Expanded Query: {expanded_query}")

        # Step 3: Graph-Based Retrieval
        graph_ranked_docs = graph_based_retrieval(processed_docs, expanded_query)

        # Step 4: Hybrid Retrieval
        hybrid_retriever = create_hybrid_retriever(processed_docs, cohere_api_key)
        if hybrid_retriever is None:
            return None
        candidate_docs = hybrid_retriever.get_relevant_documents(expanded_query)

        # Step 5: Combine and Deduplicate Candidate Documents
        combined_docs = list(
            {
                doc.page_content: doc for doc in (graph_ranked_docs + candidate_docs)
            }.values()
        )

        # Step 6: Re-Ranking using CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranked_docs = rerank_documents(expanded_query, combined_docs, cross_encoder)
        top_docs = ranked_docs[:5]

        # Step 7: Final Answer Generation
        # Combine the content of the top-ranked document chunks to create a context.
        docs_text = "\n\n".join([doc.page_content for doc in top_docs])
        final_prompt = (
            f"Based on the following context extracted from the document:\n\n{docs_text}\n\n"
            f"Answer the following question: {query_text}"
        )
        # Create a new LLM instance for answer generation.
        final_llm = Cohere(
            cohere_api_key=cohere_api_key, temperature=0.7, max_tokens=512
        )
        final_answer = final_llm(final_prompt)  # Generate the final answer.

        return {
            "result": final_answer,
            "source_documents": top_docs,
        }
    except Exception as e:
        st.error(
            f"Error generating response: {e}\n\nTraceback:\n{traceback.format_exc()}"
        )
        return None


def cleanup_chroma_db():
    """
    Clean up the persisted Chroma vector store to free up storage.
    """
    try:
        chroma_db_dir = os.path.join(os.getcwd(), "chroma_db")
        if os.path.exists(chroma_db_dir):
            shutil.rmtree(chroma_db_dir)
            st.success("ChromaDB directory cleaned up successfully.")
    except Exception as e:
        st.error(f"Error cleaning up ChromaDB directory: {e}")


# --- Streamlit App UI ---
st.title("🦜🔗 Advanced Document Query Chatbot")
st.write("## Instructions")
st.write(
    "Upload a document (txt, pdf, docx, or doc) and ask a question about its content. "
    "This version includes advanced techniques: agentic query refinement (chain-of-thought), "
    "graph-based retrieval, candidate re-ranking, query expansion, advanced document content handling, "
    "multi-step reasoning, and a comprehensive evaluation framework."
)

# File upload and query input.
uploaded_file = st.file_uploader(
    "Upload a document", type=["txt", "pdf", "docx", "doc"]
)
query_text = st.text_input(
    "Enter your question:",
    placeholder="Ask something about the document.",
    disabled=(uploaded_file is None),
)

result = None

# --- Main Query Form ---
with st.form("query_form", clear_on_submit=True):
    cohere_api_key = st.text_input(
        "Cohere API Key",
        type="password",
        help="Enter your Cohere API key (must start with x-).",
        disabled=False,
    )
    submitted = st.form_submit_button("Submit Query")
    if submitted:
        # Measure the time taken for generating the response.
        start_time = time.time()
        processed_docs = load_and_preprocess_document(uploaded_file)
        if processed_docs:
            result = generate_response(processed_docs, cohere_api_key, query_text)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"Response generated in {elapsed_time:.2f} seconds.")

if result:
    st.write("## Answer")
    st.write(result["result"])
    st.write("### Source Document Chunks Used (Re-ranked):")
    for i, doc in enumerate(result.get("source_documents", []), start=1):
        st.write(f"**Chunk {i}:**")
        st.markdown(doc.page_content)

# --- Process Advanced Content (Tables & Images) ---
advanced_content = None
if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    advanced_handler = AdvancedContentHandler()
    advanced_handler.process_document(tmp_file_path)
    os.remove(tmp_file_path)
    advanced_content = {
        "tables": advanced_handler.table_data,
        "images": advanced_handler.image_data,
    }
    if advanced_content["tables"] or advanced_content["images"]:
        st.info("Advanced content extracted from document (tables/images).")
    else:
        st.info("No advanced content detected.")

# --- Complex Query Decomposition ---
if query_text and cohere_api_key:
    if st.button("Run Complex Query Decomposition"):
        # Wrap the Cohere call in a lambda for the ComplexQueryHandler.
        complex_handler = ComplexQueryHandler(
            lambda prompt: Cohere(
                cohere_api_key=cohere_api_key, temperature=0.7, max_tokens=150
            )(prompt)
        )
        complex_result = complex_handler.process_complex_query(query_text)
        st.write("## Complex Query Handling Result")
        st.json(complex_result)

# --- Evaluation Framework Demonstration ---
if st.button("Run Evaluation (Demo)"):
    evaluator = RAGEvaluator()
    # Replace dummy test cases with realistic ones if available.
    test_cases = [
        {
            "relevant_documents": [
                "Example chunk content 1",
                "Example chunk content 2",
            ],
            "expected_answer": "The Webflow CMS plan costs 276 for 12 months.",
        }
    ]
    responses = [
        {
            "source_documents": (
                [doc.page_content for doc in result.get("source_documents", [])]
                if result
                else []
            ),
            "answer": result["result"] if result else "",
        }
    ]
    # Pass the measured elapsed time to the evaluator.
    evaluation_report = evaluator.evaluate_iteration(
        test_cases, responses #,elapsed_time
    )
    st.write("## Evaluation Report")
    st.text(evaluation_report)

st.write("----")
st.write(
    "**Note:** This prototype demonstrates advanced techniques including agentic query refinement (chain-of-thought), "
    "graph-based retrieval, candidate re-ranking, query expansion, advanced document content handling (using EasyOCR/Camelot), "
    "multi-step reasoning, and a comprehensive evaluation framework. Further enhancements (e.g., cloud-based OCR, refined answer generation) can be added in future iterations."
)

if __name__ == "__main__":
    try:
        pass
    finally:
        cleanup_chroma_db()
