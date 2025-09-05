from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")

collection = Collection("document_chunks")
collection.load()

print("Connected to Collection:", collection.name)
print("\n Schema:")
print(collection.schema)

print("\n Total entities in collection:", collection.num_entities)

print("\n Sample inserted data:")
try:
    results = collection.query(
        expr="chunk_id >= 0",
        output_fields=["chunk_id"],
        limit=2,
    )
    for r in results:
        print(r)
except Exception as e:
    print(" Query failed:", e)
