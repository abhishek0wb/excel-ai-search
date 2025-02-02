import pandas
import os, json
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Set it in .env file.")

client = OpenAI(api_key=api_key)
embedding = OpenAIEmbeddings(openai_api_key=api_key)

embedding = OpenAIEmbeddings(openai_api_type=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)   

def load_data():
    
    document = []
    text = ""
    df = pandas.read_excel(os.path.join("assets", "sample.xlsx"))

    for index, row in df.iterrows():
        raw_text = ", ".join(str(value) for value in row.values)
        document.append(raw_text)
    
    text = "\n".join(document)
        
    return text


@app.route("/vectorstore", methods=["POST"])
def vectorstore():
    data = request.json
    query = data.get("query")
    excel_data = load_data()
    print(excel_data)


    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", "!", "?", ",", " "], chunk_size=1000, chunk_overlap=250)
    characted_text = text_splitter.create_documents([excel_data])

    persistDirectory = f"Dataset/Vector_Store_900"

    vector_store = Chroma.from_documents(documents=characted_text, embedding=embedding, persist_directory=persistDirectory)

    retriever = vector_store.as_retriever(
        search_type = "similarity", search_kwargs={"k": 14}
    )

    qa = retriever.get_relevant_documents(query=query)
    output_results = []

    for results in qa:
        output_results.append({"document": results.page_content})

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
            {"role": "user", "content": []}
        ]

    conversation[1]["content"].append(
        {
            "type": "text",
            "text": query
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=conversation
    )

    return jsonify({
        "message": response.choices[0].message.content
    })

if __name__ == "__main__":
    app.run(debug=True)