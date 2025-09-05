from pymilvus import connections, utility

collection_name = "document_chunks"

connections.connect("default", host="localhost", port="19530")

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' dropped.")
else:
    print(f"Collection '{collection_name}' does not exist.")
