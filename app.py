from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient
import os

app = Flask(__name__)

class DownloadPayload:
    def __init__(self, blob_name):
        self.blob_name = blob_name

def read_text_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        return None


def download_file_from_blob(container_name, blob_name):
    try:
        connection_string = "DefaultEndpointsProtocol=https;AccountName=azuretestshubham832458;AccountKey=2yEaP59qlgKVv6kEUCA5ARB4wdV3ZRoL2X9zjYCcIxOSYAG1CSBbBlAMPx3uBIe7ilQtSh7purEK+AStvFn8GA==;EndpointSuffix=core.windows.net"  # Replace with your Azure Blob Storage connection string
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        destination_path = f"transcripts/{blob_name}"  # Replace with the desired destination path
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)  # Create the destination directory if it doesn't exist
        
        with open(destination_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        
        print(f"File downloaded successfully: {destination_path}")
        return True
    except Exception as e:
        print(f"Error downloading file '{blob_name}': {str(e)}")
        return False


app = Flask(__name__)

@app.route("/download", methods=["POST"])
def download_file():
    payload = request.json
    blob_name = payload.get("blob_name")
    
    if not blob_name:
        return jsonify({"message": "Blob name not provided."}), 400
    
    success = download_file_from_blob("transcript", blob_name)
    
    if success:
        return jsonify({"message": "File download completed successfully."})
    else:
        return jsonify({"message": "File download failed."}), 500




if __name__ == "__main__":
    app.run()
