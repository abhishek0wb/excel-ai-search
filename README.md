# Excel AI SEARCH

A Flask-based API that enables semantic search over Excel data using OpenAI embeddings and ChromaDB vector store.

## Overview

This application provides a RESTful API endpoint that:
1. Loads data from an Excel file
2. Converts the data into vector embeddings using OpenAI's embedding model
3. Stores these embeddings in a ChromaDB vector store
4. Performs semantic similarity search based on user queries
5. Uses GPT-4 to generate natural language responses based on the retrieved context


## Environment Setup

Create a `.env` file in the root directory with your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_api_key_here
```

## Running Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd excel-ai-search-main
    ```

2.  **Create and activate a virtual environment:**
    On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: See the "Common Issues" section below if you encounter installation errors on Windows.*

4.  **Set up environment variables:**
    Create a `.env` file as described in the [Environment Setup](#environment-setup) section.

5.  **Run the application:**
    ```bash
    flask run
    ```
    The API will be available at `http://127.0.0.1:5000`.

## Common Issues

### Microsoft Visual C++ Redistributable Requirement

When installing dependencies via `pip` on a Windows machine, you may encounter an error message indicating that Microsoft Visual C++ 14.0 or greater is required. This is because some Python packages, like `hnswlib` (a dependency for ChromaDB), need to be compiled from source and require a C++ toolchain.

**Solution:**
Download and install the "Build Tools for Visual Studio" from the official [Microsoft Visual Studio website](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, make sure to select the "C++ build tools" workload. After the installation is complete, try running `pip install -r requirements.txt` again.


## API Endpoints

### POST /vectorstore

Performs semantic search on the Excel data and returns AI-generated responses.

**Request Body:**
```json
{
    "query": "Your search query here"
}
```

**Response:**
```json
{
    "message": "AI-generated response based on the relevant context"
}
```

## Excel File Requirements

Place your Excel file in the `assets` folder with the name `sample.xlsx`. The code will process all columns in the Excel file and combine them into searchable text.

## Vector Store Configuration

The application uses the following configuration for text splitting and vector search:
- Chunk size: 1000 characters
- Chunk overlap: 250 characters
- Search type: Similarity search
- Number of similar documents retrieved: 14
