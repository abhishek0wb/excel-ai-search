import os
import json
import pandas
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables
load_dotenv()

# Retrieve API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env file.")

# Initialize OpenAI client and embedding
client = OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize Flask app
app = Flask(__name__)

def load_data():
    """
    Loads and processes data from an Excel file into a single text format.
    """
    document = []
    df = pandas.read_excel(os.path.join("assets", "sample.xlsx"))

    for _, row in df.iterrows():
        raw_text = ", ".join(str(value) for value in row.values)
        document.append(raw_text)

    return "\n".join(document)


@app.route("/vectorstore", methods=["POST"])
def vectorstore():
    """
    Handles user queries, retrieves relevant documents from ChromaDB, 
    and generates responses using OpenAI's API.
    """
    # Validate request input
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing or empty 'query' parameter"}), 400

    # Load and preprocess the Excel data
    excel_data = load_data()
    print(excel_data)

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "], 
        chunk_size=1000, 
        chunk_overlap=250
    )
    character_text = text_splitter.create_documents([excel_data])

    persist_directory = "Dataset/Vector_Store_900"

    # Create ChromaDB vector store
    vector_store = Chroma.from_documents(
        documents=character_text, 
        embedding=embedding, 
        persist_directory=persist_directory
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 14})

    # Retrieve relevant documents
    qa = retriever.get_relevant_documents(query=query)
    output_results = [{"document": result.page_content} for result in qa]

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
        {
            "role": "user",
            "content": [{"type": "text", "text": query}]
        }
    ]

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=conversation
    )

    return jsonify({"message": response.choices[0].message.content})


if __name__ == "__main__":
    app.run(debug=True)
