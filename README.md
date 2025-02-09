# Ooredoo AI Engineer Interview Project: Advanced Document Query Chatbot

This project demonstrates the innovative use of AI to enhance document querying, showcasing how advanced RAG techniques can transform document interaction in a business context. It goes beyond basic Retrieval-Augmented Generation to incorporate cutting-edge methods for improved accuracy, relevance, and handling of complex document types.

## Title: AI-Powered Document Query Chatbot with Advanced RAG

This project is a demonstration application built for the Ooredoo AI Engineer interview process. It showcases the development of a sophisticated chatbot capable of answering user queries based on uploaded documents, leveraging **Retrieval-Augmented Generation (RAG)** and a suite of **advanced retrieval techniques**.

## Interview Task Alignment

This project directly addresses and exceeds the requirements outlined in the Ooredoo AI Engineer interview task, specifically:

*   **Building a Chatbot Application:** The core of this project is a fully functional chatbot application with a user-friendly Streamlit interface, ready for immediate demonstration.
*   **Gen AI and Retrieval-Augmented Generation (RAG):** The application is built upon Generative AI principles, implementing a comprehensive RAG pipeline to answer user questions grounded in provided documents.
*   **Advanced RAG Techniques (Extensive Implementation):**  The application demonstrably goes beyond basic RAG by incorporating a range of advanced techniques to address complex queries and improve retrieval effectiveness. These include:
    *   **Hybrid Retrieval:** Combining vector-based semantic search with keyword-based BM25 retrieval for enhanced recall and precision.
    *   **Query Expansion:** Utilizing WordNet to broaden search queries and capture vocabulary variations.
    *   **Chain-of-Thought Query Refinement:** Employing a Cohere LLM to refine user queries for optimized document retrieval.
    *   **Graph-Based Retrieval:** Leveraging named entity recognition and graph algorithms to rank document chunks based on semantic relationships.
    *   **Re-ranking:** Utilizing a CrossEncoder model to re-rank retrieved documents for improved contextual relevance.
    *   **Multi-Step Reasoning and Query Decomposition:** Implementing an agentic approach to break down complex queries into sub-questions for more effective answer generation.
*   **Advanced Document Content Handling:**  The application features a robust preprocessing pipeline capable of handling various document types (PDF, DOCX, TXT, DOC) and, crucially, extracting content from:
    *   **Tables (PDF Documents):** Utilizing Camelot to extract tabular data from PDF files.
    *   **Images (Within Documents):**  Implementing EasyOCR to extract text from images embedded within documents, making image-based information searchable.
*   **Presentation-Ready Project:** The codebase is meticulously structured, extensively commented, and includes a functional Streamlit UI, making it ideal for a detailed code walkthrough and live demonstration.
*   **Comprehensive Evaluation Framework:**  The project includes a built-in evaluation framework with quantifiable metrics (Precision, Recall, Semantic Similarity) to assess RAG system performance and demonstrate iterative improvement.

## Key Features

*   **Versatile Document Loading:**  Handles multiple document formats (`.txt`, `.pdf`, `.docx`, `.doc`) using Langchain's `UnstructuredFileLoader`.
*   **Advanced Document Preprocessing:**  Documents are parsed, cleaned, and intelligently chunked using Langchain's `RecursiveCharacterTextSplitter` for semantic context retention.
*   **Cutting-Edge Hybrid Retrieval:** Employs `EnsembleRetriever` to synergistically combine:
    *   **Vector Search (ChromaDB):**  Semantic similarity search powered by high-quality Cohere Embeddings for deep contextual understanding.
    *   **BM25 Retrieval:**  Keyword-based retrieval using BM25 for robust lexical matching, ensuring comprehensive recall.
*   **Sophisticated Query Enhancement:**  Implements:
    *   **Query Expansion:**  Leveraging NLTK's WordNet to expand user queries with synonyms, significantly improving recall.
    *   **Chain-of-Thought Query Refinement:**  Utilizing a Cohere LLM to intelligently refine user queries, optimizing them for effective document retrieval through chain-of-thought reasoning.
*   **Innovative Graph-Based Retrieval:**  Ranks document chunks based on shared named entities using spaCy and NetworkX's PageRank algorithm, capturing semantic relationships and contextual relevance.
*   **Contextual Re-ranking:**  Employs a Sentence Transformers CrossEncoder model to re-rank candidate documents, prioritizing those most contextually relevant to the expanded and refined query.
*   **Multi-Step Reasoning for Complex Queries:**  Features a `ComplexQueryHandler` that decomposes complex user questions into a series of sub-queries, enabling the chatbot to tackle nuanced and multi-faceted inquiries through a step-by-step reasoning process.
*   **Advanced Document Content Extraction:**  Integrates:
    *   **Camelot:** For extracting tabular data from PDF documents, making structured information accessible to the chatbot.
    *   **EasyOCR:** For Optical Character Recognition, enabling the chatbot to extract and understand text from images within documents, broadening the scope of searchable content.
*   **Intelligent Conversational Question Answering:**  Leverages Cohere's powerful LLM within a Langchain-based RAG pipeline to generate insightful, contextually accurate, and comprehensive answers.
*   **Interactive and User-Friendly Chatbot UI:**  A streamlined and intuitive interface built with Streamlit, providing:
    *   Effortless document upload functionality with clear file type and size limits.
    *   Direct text input for intuitive user queries.
    *   Real-time streaming answer display, enhancing user engagement.
    *   Transparent visualization of re-ranked source document chunks, clearly showing the context used for answer generation.
    *   Integrated basic evaluation section, allowing for direct testing and demonstration of performance metrics.
*   **Robust Error Handling and User Feedback:**  Incorporates comprehensive error handling throughout the application, providing informative feedback messages within the Streamlit UI to ensure a smooth and user-friendly experience.
*   **Persistent and Efficient Vector Database:**  Utilizes ChromaDB for efficient vector storage and similarity search, with persistence enabled to maintain the vector index between sessions.
*   **Comprehensive Evaluation Framework:** Includes a dedicated `RAGEvaluator` class that calculates quantifiable metrics (Precision, Recall, Semantic Similarity) to systematically assess and demonstrate the RAG system's performance and track improvements.

## Project Structure

*   `advanced_streamlit_app.py`: Main Python application file containing the Streamlit UI, backend logic, and advanced RAG pipeline implementation.
*   `requirements.txt`:  Detailed list of all Python dependencies required to run the application, ensuring easy environment setup.
*   `chroma_db/`: Directory for persistent ChromaDB vector database storage, ensuring data is preserved across sessions.

## Setup

### Prerequisites

*   Python 3.8 or higher
*   [Streamlit](https://streamlit.io/) (installed via `requirements.txt`)
*   [Langchain](https://python.langchain.com/) (installed via `requirements.txt`)
*   [ChromaDB](https://www.trychroma.com/) (installed via `requirements.txt`)
*   [Sentence Transformers](https://www.sbert.net/) (installed via `requirements.txt`)
*   [spaCy](https://spacy.io/) and `en_core_web_sm` model (installed via `requirements.txt`)
*   [NetworkX](https://networkx.org/) (installed via `requirements.txt`)
*   [NLTK](https://www.nltk.org/) and WordNet corpus (installed via `requirements.txt` and code)
*   [OpenCV](https://opencv.org/) (`cv2` - installed via `requirements.txt`)
*   [EasyOCR](https://www.jaided.ai/easyocr/) (installed via `requirements.txt`)
*   [Camelot-py](https://camelot-py.readthedocs.io/en/master/) (installed via `requirements.txt`)
*   Cohere API Key (Obtain from [Cohere AI](https://cohere.ai/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KatoStevenMubiru/Ooredoo-RAG
    cd Ooredoo-RAG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv ooredoo_rag_env  # Or conda create -n ooredoo_rag_env python=3.9
    source ooredoo_rag_env/bin/activate  # Or conda activate ooredoo_rag_env
    ```

3.  **Install all Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy English Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up your Cohere API key:**
    *   Obtain a Cohere API key from [Cohere AI](https://cohere.ai/).
    *   Set the API key as an environment variable (recommended for security):
        ```bash
        export COHERE_API_KEY="YOUR_COHERE_API_KEY"
        ```
        *Replace `"YOUR_COHERE_API_KEY"` with your actual Cohere API key.*
    *   Alternatively, you can directly input your API key into the Streamlit app when prompted (less secure for sharing code).

6.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Evaluation

*   **Evaluation Framework:**  The project includes a built-in `RAGEvaluator` class that automatically calculates:
    *   **Retrieval Precision:** Measures the accuracy of document retrieval.
    *   **Retrieval Recall:** Measures the completeness of document retrieval.
    *   **Semantic Similarity:**  Measures the semantic similarity between the chatbot's answer and an expected answer, reflecting answer quality.
*   **Demonstration:** The Streamlit UI includes a section to run a basic evaluation demo and display the generated evaluation report, showcasing the framework in action.
*   **Future Enhancements:** Future iterations will focus on expanding the evaluation framework with more comprehensive metrics (e.g., Faithfulness, Answer Relevance) and creating more robust test datasets for rigorous performance assessment.

## Contributing

Contributions to enhance this project are highly welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to fork this repository and submit pull requests.

## Future Work

*   **Enhanced Document Processing:**
    *   Further refinement of table extraction and indexing for improved handling of structured data.
    *   Integration of cloud-based OCR services for more robust and scalable image text extraction.
    *   Advanced processing of extracted table and image content to enable deeper semantic understanding and querying.
*   **Advanced RAG Techniques:**
    *   Implementation of Context Compression techniques for improved efficiency and reduced noise in retrieved context.
    *   Exploration of more sophisticated Agentic RAG workflows for complex reasoning and multi-turn conversations.
*   **Evaluation Framework Expansion:**
    *   Development of more comprehensive and nuanced evaluation metrics, such as Faithfulness and Answer Relevance, potentially using LLM-based evaluation methods.
    *   Creation of more diverse and challenging test datasets to rigorously assess the chatbot's performance across various query types and document complexities.
*   **Production Readiness and Scalability:**
    *   Migration to a cloud-based vector database for improved scalability and performance in production environments.
    *   Implementation of API key security best practices and secure handling of sensitive information.
    *   Integration of user authentication and authorization mechanisms for secure access control.
    *   Exploration of and integration with more diverse LLM providers to offer users a choice of models and enhance system flexibility.
*   **Multi-turn Conversational Interface:**  Development of a more advanced conversational UI to support multi-turn dialogues, follow-up questions, and more interactive user experiences.

## License

[MIT License](LICENSE)  (Add a LICENSE file to your repository if you choose to use the MIT license)

## Acknowledgments

*   Built using the powerful [Langchain](https://python.langchain.com/) framework for Retrieval-Augmented Generation.
*   Leverages the efficient and user-friendly [ChromaDB](https://www.trychroma.com/) for vector database capabilities.
*   Utilizes the high-performance language models and embeddings provided by [Cohere AI](https://cohere.ai/).
*   Thanks to the [Sentence Transformers](https://www.sbert.net/) library for efficient semantic embeddings and CrossEncoder models.
*   Utilizes [spaCy](https://spacy.io/) for advanced Natural Language Processing tasks, particularly Named Entity Recognition.
*   Employs [NetworkX](https://networkx.org/) for graph-based data structures and algorithms.
*   Leverages [NLTK](https://www.nltk.org/) and WordNet for lexical resources and query expansion.
*   Integrates [OpenCV](https://opencv.org/) (`cv2`) for robust image processing.
*   Incorporates [EasyOCR](https://www.jaided.ai/easyocr/) for accessible Optical Character Recognition.
*   Utilizes [Camelot-py](https://camelot-py.readthedocs.io/en/master/) for effective PDF table extraction.