import os
import pandas
import json
import logging
import time
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env file.")

# Initialize OpenAI client and embedding
client = OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

# Set up logging for error tracking
logging.basicConfig(filename="app_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

def load_data():
    """
    Loads data from an Excel file and processes it into a single text format.
    Handles file-related errors gracefully.
    """
    document = []
    
    try:
        df = pandas.read_excel(os.path.join("assets", "sample.xlsx"))
    except FileNotFoundError:
        logging.error("Excel file not found in assets directory.")
        raise FileNotFoundError("Excel file not found. Please ensure 'sample.xlsx' is in the 'assets' directory.")
    except Exception as e:
        logging.error(f"Error loading Excel file: {str(e)}")
        raise RuntimeError(f"Error loading Excel file: {str(e)}")

    for _, row in df.iterrows():
        raw_text = ", ".join(str(value) for value in row.values)
        document.append(raw_text)
    
    return "\n".join(document)

def call_openai_with_retry(conversation, max_retries=3, timeout=10):
    """
    Calls OpenAI API with retries and timeout handling.
    Prevents unnecessary failures due to temporary network issues.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=conversation,
                timeout=timeout  # Setting a timeout for API call
            )
        except OpenAIError as e:
            logging.error(f"OpenAI API Error: {str(e)} - Attempt {attempt+1} of {max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff before retrying
            else:
                raise OpenAIError(f"OpenAI API failed after {max_retries} attempts: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected OpenAI API error: {str(e)}")
            raise RuntimeError("Unexpected error while querying OpenAI API.")

@app.route("/vectorstore", methods=["POST"])
def vectorstore():
    """
    Handles user queries, retrieves relevant documents from ChromaDB, and uses OpenAI API to generate a response.
    Implements error handling for missing inputs, OpenAI failures, and vector store issues.
    """
    try:
        # Validate request input
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' parameter in request."}), 400
        
        query = data.get("query").strip()
        if not query:
            return jsonify({"error": "Query cannot be empty."}), 400

        # Load and preprocess the Excel data
        excel_data = load_data()

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", ",", " "], 
            chunk_size=1000, 
            chunk_overlap=250
        )
        characted_text = text_splitter.create_documents([excel_data])

        persistDirectory = "Dataset/Vector_Store_900"

        # Create ChromaDB vector store with error handling
        try:
            vector_store = Chroma.from_documents(
                documents=characted_text, 
                embedding=embedding, 
                persist_directory=persistDirectory
            )
        except Exception as e:
            logging.error(f"ChromaDB initialization error: {str(e)}")
            return jsonify({"error": "Failed to initialize vector store. Please check logs."}), 500

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 14}
        )

        # Retrieve relevant documents
        qa = retriever.get_relevant_documents(query=query)
        output_results = [{"document": results.page_content} for results in qa]

        # If no relevant data found, return an appropriate response
        if not output_results:
            return jsonify({"message": "No relevant information found in the dataset."}), 200

        # Construct conversation prompt
        conversation = [
            {
                "role": "system",
                "content": f"""
                You are a helpful AI assistant with access to the following context from an Excel file:

                {output_results}

                When answering the user's question, you must:
                - Base your response solely on the context above.
                - If the context does not contain enough information to answer fully, say so.
                """,
            },
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ]

        # Call OpenAI API with retry mechanism
        try:
            response = call_openai_with_retry(conversation)
        except OpenAIError as e:
            logging.error(f"Final OpenAI API failure: {str(e)}")
            return jsonify({"error": "AI service is currently unavailable. Please try again later."}), 500
        except Exception as e:
            logging.error(f"Unexpected error in AI response: {str(e)}")
            return jsonify({"error": "An unexpected error occurred while generating the response."}), 500

        return jsonify({"message": response.choices[0].message.content})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logging.error(f"Unhandled error in /vectorstore route: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please check the server logs."}), 500

if __name__ == "__main__":
    app.run(debug=True)
