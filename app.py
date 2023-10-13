from flask import Flask, request, jsonify
import os
import tempfile
import json
from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone

app = Flask(__name__)

temp_directory = tempfile.mkdtemp()
directory = temp_directory

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

pinecone.init(
    api_key="af24bb75-2108-433c-84d4-18a573c00d0b",
    environment="gcp-starter"
)

index_name = "example-index"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

os.environ["OPENAI_API_KEY"] = "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"
model_name = "gpt-3.5-turbo"
llm = OpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(query):
    similar_docs = get_similiar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

connection_string = "DefaultEndpointsProtocol=https;AccountName=azuretestshubham832458;AccountKey=2yEaP59qlgKVv6kEUCA5ARB4wdV3ZRoL2X9zjYCcIxOSYAG1CSBbBlAMPx3uBIe7ilQtSh7purEK+AStvFn8GA==;EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)

@app.route('/process_blob', methods=['POST'])
def process_blob():
    try:
        req_data = request.get_json()
        container_name = req_data.get('container_name')
        blob_name = req_data.get('blob_name')
        query = req_data.get('query')
        api_key = request.headers.get('X-API-Key')
        os.environ["OPENAI_API_KEY"] = api_key

        answer = get_answer(query)

        if not container_name or not blob_name:
            return jsonify({"error": "Please provide both 'container_name' and 'blob_name' in the request JSON"}), 400

        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        transcript_directory = os.path.join(temp_directory, 'transcript')
        os.makedirs(transcript_directory, exist_ok=True)

        local_file_path = os.path.join(transcript_directory, blob_name)

        with open(local_file_path, "wb") as local_file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(local_file)

        response_data = {
            "download_message": f"Blob {blob_name} has been downloaded to {local_file_path}",
            "answer": answer
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
