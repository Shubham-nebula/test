from flask import Flask, request, jsonify
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

# Load documents and create a vector store
directory = 'transcript'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)
persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()

# Define a function to set the API key
def set_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

# Load the question answering chain and initialize the model inside the API
@app.route('/answer', methods=['POST'])
def get_answer():
    try:
        # Get the query from the request body
        query = request.json['query']

        # Get the API key from the request headers
        api_key = request.headers.get('X-API-Key')

        # Set the API key using os.environ
        set_api_key(api_key)

        # Load the language model inside the API
        from langchain.chat_models import ChatOpenAI
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name)

        # Load the question answering chain
        chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

        # Search for matching documents
        matching_docs = db.similarity_search(query)

        # Run the question answering chain
        answer = chain.run(input_documents=matching_docs, question=query)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
