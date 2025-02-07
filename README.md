# Ooredoo AI Engineer Interview Project: Advanced Document Query Chatbot

This project demonstrates the innovative use of AI to enhance document querying, showcasing how advanced RAG techniques can transform document interaction in a business context.

## Title: AI-Powered Document Query Chatbot with Advanced RAG

This project is a demonstration application built for the Ooredoo AI Engineer interview process. It showcases the development of a sophisticated chatbot capable of answering user queries based on uploaded documents, leveraging **Retrieval-Augmented Generation (RAG)** and **advanced retrieval techniques**.

## Interview Task Alignment

This project directly addresses the requirements outlined in the Ooredoo AI Engineer interview task, specifically:

*   **Building a Chatbot Application:** The core of this project is a functional chatbot application with a user-friendly interface.
*   **Gen AI and Retrieval-Augmented Generation (RAG):** The application is built upon Gen AI principles and implements a RAG pipeline to answer questions from documents.
*   **Advanced RAG Techniques:**  The application goes beyond basic RAG by incorporating **Hybrid Retrieval**, combining vector-based semantic search with keyword-based BM25 retrieval for enhanced accuracy and relevance.
*   **Document Preprocessing:**  The application includes a robust preprocessing pipeline capable of handling various document types (PDF, DOCX, TXT, DOC) and efficiently chunking documents for optimal retrieval.
*   **Presentation-Ready Project:** The codebase is structured, well-commented, and includes a functional Streamlit UI, ready for code walkthrough and demonstration.
*   **Evaluation Awareness:** The project includes a basic evaluation framework and demonstrates an understanding of the importance of evaluation in RAG system development.

## Key Features

*   **Versatile Document Loading:**  Handles multiple document formats (`.txt`, `.pdf`, `.docx`,`.doc`) using Langchain's `UnstructuredFileLoader`.
*   **Efficient Document Preprocessing:**  Documents are parsed, cleaned, and chunked using Langchain's `RecursiveCharacterTextSplitter` for semantic awareness.
*   **Advanced Hybrid Retrieval:** Implements `EnsembleRetriever` to combine:
    *   **Vector Search (ChromaDB):** Semantic similarity search using Cohere Embeddings for contextual understanding.
    *   **BM25 Retrieval:** Keyword-based retrieval for lexical matching, enhancing recall.
*   **Conversational Question Answering:**  Utilizes Cohere's powerful LLM within a Langchain `RetrievalQA` chain to generate insightful and contextually relevant answers.
*   **Interactive Chatbot UI:**  A user-friendly interface built with Streamlit, featuring:
    *   Document upload functionality.
    *   Text input for user queries.
    *   Display of AI-generated answers with a streaming typing effect.
    *   Visualization of source document chunks used for answer generation, enhancing transparency.
    *   Basic evaluation section for testing and demonstrating performance.
*   **Error Handling and User Feedback:**  Includes robust error handling and informative feedback messages within the Streamlit application to improve user experience.
*   **Persistent Vector Database:**  Leverages ChromaDB with persistence to store document embeddings efficiently.

## Project Structure

*   `advanced_streamlit_app.py`: Main application file for the Streamlit UI and backend logic.
*   `requirements.txt`: List of Python dependencies.
*   `chroma_db/`: Directory for ChromaDB persistence.

## Setup

### Prerequisites

*   Python 3.8 or higher
*   [Streamlit](https://streamlit.io/)
*   [Langchain](https://python.langchain.com/)
*   [ChromaDB](https://www.trychroma.com/)
*   Cohere API Key (Obtain from [Cohere](https://cohere.ai/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd [repository-directory]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv gemini_env  # Or conda create -n gemini_env python=3.9
    source gemini_env/bin/activate  # Or conda activate gemini_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Cohere API key:**
    *   Obtain a Cohere API key from [Cohere](https://cohere.ai/).
    *   Set the API key as an environment variable (recommended for security):
        ```bash
        export COHERE_API_KEY="YOUR_COHERE_API_KEY"
        ```
        *Replace `"YOUR_COHERE_API_KEY"` with your actual Cohere API key.*
    *   Alternatively, you can directly input your API key into the Streamlit app when prompted (less secure for sharing code).

5.  **Run the Application:**
    ```bash
    streamlit run advanced_streamlit_app.py
    ```

## Evaluation

*   **Evaluation Metrics:** Currently, we use manual comparison against expected answers. Future iterations could include automated metrics such as precision, recall, or user satisfaction ratings.

## Contributing

Contributions are welcome! If you have ideas for improvements or want to add new features, please fork this repository and submit pull requests.

## Future Work

*   Enhance image processing capabilities with OCR for document images.
*   Improve table parsing for better handling of structured data within documents.
*   Implement more advanced RAG techniques such as graph-based retrieval or multi-step reasoning.
*   Optimize performance with query caching or by using more efficient embedding models.

## License

 MIT License

## Acknowledgments

*   Thanks to [Langchain](https://python.langchain.com/) for the RAG framework.
*   Special thanks to [ChromaDB](https://www.trychroma.com/) for vector database capabilities.
*   Appreciation to [Cohere](https://cohere.ai/) for their language model services.
