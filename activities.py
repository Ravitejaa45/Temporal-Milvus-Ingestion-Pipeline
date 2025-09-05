import aiohttp
import os
import uuid
import tempfile
from typing import List, Dict

from unstructured.partition.auto import partition
from pymilvus import connections, Collection, utility, DataType, CollectionSchema, FieldSchema
from temporalio import activity

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = tempfile.gettempdir()

# Choose a HF sentence embedding model via env var
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/e5-large-v2")

_sbert = SentenceTransformer(HF_EMBED_MODEL)
EMBED_DIM = _sbert.get_sentence_embedding_dimension()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

collection_name = "document_chunks"

if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description=f"Document chunks with {HF_EMBED_MODEL} embeddings ({EMBED_DIM}d)")
    Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created with dim={EMBED_DIM}.")
else:
    existing = Collection(name=collection_name)
    for f in existing.schema.fields:
        if f.name == "embedding" and getattr(f.params, "dim", EMBED_DIM) != EMBED_DIM:
            print(f" Milvus collection '{collection_name}' has dim={getattr(f.params, 'dim', 'unknown')} "
                  f"but model produces {EMBED_DIM}d. Consider dropping the collection or using a new one.")
collection = Collection(name=collection_name)
if not any(getattr(idx, "field_name", None) == "embedding" for idx in collection.indexes):
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "IP"
        }
    )
    print(f" Index created on 'embedding' field for collection '{collection_name}'")

collection.load()
print(f"Collection '{collection_name}' loaded")

@activity.defn
async def fetch_document(file_url: str, file_id: str) -> str:
    try:
        filename = f"{file_id}_{uuid.uuid4().hex[:6]}.pdf"
        filepath = os.path.join(TEMP_DIR, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download file: {resp.status}")
                with open(filepath, "wb") as f:
                    f.write(await resp.read())
        return filepath
    except aiohttp.ClientError as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

@activity.defn
def parse_document(filepath: str) -> List[str]:
    try:
        elements = partition(filename=filepath)
        return [el.text.strip() for el in elements if getattr(el, "text", None) and el.text.strip()]
    except Exception as e:
        raise Exception(f"Parsing failed: {str(e)}")

@activity.defn
def generate_embeddings(chunks: List[str]) -> List[List[float]]:

    if not chunks:
        return []
    try:
        to_encode = (
            [f"passage: {c}" for c in chunks]
            if "e5" in HF_EMBED_MODEL.lower()
            else chunks
        )
        vectors = _sbert.encode(
            to_encode,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
    except Exception as e:
        print(f" Error while generating embeddings: {str(e)}")
        return []

@activity.defn
def store_in_milvus(vectors: List[List[float]]) -> Dict:
    try:
        if not vectors:
            raise ValueError("No vectors to insert into Milvus")
        collection.insert([vectors])
        collection.flush()
        return {"inserted": len(vectors), "collection": collection_name, "dim": EMBED_DIM}
    except Exception as e:
        raise Exception(f"Milvus insert error: {str(e)}")
