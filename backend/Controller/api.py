import os
import json
import pandas
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env file.")

client = OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_directory = "Dataset/Vector_Store_900"

def create_vector_store():
    """
    Loads data from Excel, splits it, and creates a persistent vector store.
    This function should be run once, or whenever the Excel file changes.
    """
    print("Creating vector store...")
    document = []
    df = pandas.read_excel(os.path.join("assets", "sample.xlsx"))

    for _, row in df.iterrows():
        raw_text = ", ".join(str(value) for value in row.values)
        document.append(raw_text)

    full_text = "\n".join(document)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "], 
        chunk_size=1000, 
        chunk_overlap=250
    )
    character_text = text_splitter.create_documents([full_text])

    # Create and persist the vector store
    Chroma.from_documents(
        documents=character_text, 
        embedding=embedding, 
        persist_directory=persist_directory
    )
    print("Vector store created successfully.")
    return {"status": "success", "message": f"Vector store created at {persist_directory}"}


def handle_query(query):
    """
    Handles an incoming query by loading the persistent vector store
    and using it to answer the question.
    """
    # Check if the vector store exists, if not, create it.
    if not os.path.exists(persist_directory):
        create_vector_store()

    # Now, load the persisted vector store
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 14})

    # Use the new 'invoke' method instead of the deprecated one
    qa = retriever.invoke(query)
    output_results = [{"document": result.page_content} for result in qa]

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

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=conversation
    )

    return {"message": response.choices[0].message.content}

# You can add a small block to run the creation script directly
if __name__ == '__main__':
    create_vector_store()