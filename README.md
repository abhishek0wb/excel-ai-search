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
