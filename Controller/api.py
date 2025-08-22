import os
import json
import pandas
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env file.")

client = OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

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

def handle_query(query):
    excel_data = load_data()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "], 
        chunk_size=1000, 
        chunk_overlap=250
    )
    character_text = text_splitter.create_documents([excel_data])

    persist_directory = "Dataset/Vector_Store_900"

    vector_store = Chroma.from_documents(
        documents=character_text, 
        embedding=embedding, 
        persist_directory=persist_directory
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 14})

    qa = retriever.get_relevant_documents(query=query)
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
