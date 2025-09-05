# Temporal + Milvus Ingestion Pipeline

Process a document from a URL → parse into chunks → generate sentence embeddings → store text + vectors in Milvus - all orchestrated by Temporal with async/await concurrency.

### All About this Pipeline:

- Temporal **Workflow** + **Activities** in Python (async).
  
- High-throughput concurrency with  `workflow.start_activity(...)` `asyncio.gather(...)`.
  
- Robust retries, backoff, and graceful failure for unsupported types.

- Milvus schema storing: `file_id`, `chunk_index`, `chunk_text`, `embedding`.

## Technical Stack Summary

| Layer             | Tools / Libraries                                      |
|------------------|--------------------------------------------------------|
| Workflow Engine | Temporal (Server + UI via Docker) |
| Python SDK        | `temporalio`|
| Parser     | Unstructured  (`unstructured.partition.auto.partition`) |
| Embeddings       | Hugging Face `sentence-transformers`  (default: `intfloat/e5-large-v2`)|
| Vector DB   | Milvus (`pymilvus`)|
| I/O         | `aiohttp` (async downloads)|
| Orchestration| `asyncio` |

Supported file types: **.docx, .doc, .pdf, .xlsx, .xls**


## Workflow and Activities Structure

- **Workflow:** `DocumentIngestionWorkflow.run(file_id, file_url)`
  
  1. `fetch_document` - downloads the file with `aiohttp` and saves it with the original suffix (`.pdf/.docx/.xlsx/...`)  to help Unstructured pick the correct loader.

  2. `parse_document` - uses `unstructured.partition.auto.partition`  to produce clean text chunks.

  3. `generate_embeddings` - creates normalized embeddings with a sentence-transformer (default `intfloat/e5-large-v2` ).

  4. `store_in_milvus` - inserts `file_id`, `chunk_index`, `chunk_text`, `embedding` into Milvus.

- The workflow returns a compact JSON including a **small sample preview** (first 2 chunk texts + first 3 floats of their embeddings).

## Asyncio Concurrency

- **Parallel embedding & upserts:**
  
  - Records are **batched** (`EMBED_BATCH_SIZE`, `UPSERT_BATCH_SIZE`).

  - For each window, the workflow launches multiple activities **in parallel** via `workflow.start_activity(...)` and awaits them with `asyncio.gather(...)`.

- **Throughput controls** via `EMBED_CONCURRENCY`, `UPSERT_CONCURRENCY` env vars.

- **Worker** uses a `ThreadPoolExecutor` so any CPU-bound parts don’t block the event loop, while the workflow stays async.

## Error Handling & Retries

- **Unsupported types** are rejected early with a **non-retryable** `ApplicationError`  (allowed:  `.docx, .doc, .pdf, .xlsx, .xls`).

- Each activity call specifies a **RetryPolicy** (exponential backoff, capped attempts) so transient network/DB errors are retried by Temporal.

- Activities wrap their own logic in `try/except`    and surface clear error messages (download/parse/embed/Milvus insert).

- **Sanity checks** (e.g., embedding count == chunk count per batch) raise `ApplicationError`  to fail fast on logic/data mismatches.

## Milvus Schema

- **Collection:** `document_chunks`

- **Fields:**

    - `chunk_id` - `INT64`, primary key, `auto_id=True`

    - `file_id` - `VARCHAR(100)`

    - `chunk_index` - `INT64`

    - `chunk_text` - `VARCHAR(2000)`

    - `embedding` - `FLOAT_VECTOR(dim=<model_dim>)` (e.g., 1024 for `e5-large-v2`)

- **Index:** `AUTOINDEX` on `embedding`, `metric_type=IP` (cosine via normalized vectors)


## Assumptions

- **Embedding provider:** Using Hugging Face locally (no external keys). The embedding layer is isolated, so you can swap to OpenAI with minimal changes if required.

- **Dempotency:** Current demo appends rows (auto PK). For strict idempotency, use a deterministic PK (e.g., hash of `file_id + chunk_index`) or implement upsert semantics.

- **Security:** URLs are assumed to be safe/public. In production, add validation and sandboxing for downloads.

- **Chunking:** Default Unstructured partitioning. You can add overlap/semantic chunking without changing the workflow shape.

## Project Structure

```text
Ingestion Pipeline/
├── activities.py
├── workflows.py
├── worker.py
├── client.py       
├── inspect_milvus.py                                        
├── drop_collection.py
├── docker-compose.yml
├── requirements.txt 
├── .env
└── README.md             
```

# Setup Instructions

## Prereqs

- Docker Desktop (with `docker compose`)

## 1. Clone the repository

```bash
git clone <path to the git code>
cd Temporal-Milvus-Ingestion-Pipeline
```

## 2. Create & activate venv

```bash
python -m venv <env_name>
venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 4. Set your (env vars)

Set in your environment or a `.env` file:

 | Variable             |                Default | Description                                 |
| -------------------- | ---------------------: | ------------------------------------------- |
| `HF_EMBED_MODEL`     | `intfloat/e5-large-v2` | Hugging Face sentence embedding model       |
| `MILVUS_HOST`        |            `localhost` | Milvus host                                 |
| `MILVUS_PORT`        |                `19530` | Milvus port                                 |
| `EMBED_BATCH_SIZE`   |                   `64` | Chunk texts per **embedding** activity call |
| `UPSERT_BATCH_SIZE`  |                  `128` | Rows per **Milvus insert** activity call    |
| `EMBED_CONCURRENCY`  |                    `8` | Number of concurrent embedding batches      |
| `UPSERT_CONCURRENCY` |                    `8` | Number of concurrent upsert batches         |

Embeddings are L2-normalized; Milvus index uses **Inner Product (IP)** so cosine ~ IP.


## 5. Reset any old Milvus schema (Optional)

```bash
python drop_collection.py
```

## 6. Start infra (Milvus, Temporal, UI, Postgres)

```bash
docker compose up -d
docker ps
```

You should see containers: `milvus-standalone`, `milvus-etcd`, `milvus-minio`, `temporal`, `temporal-ui`, `temporal-postgres`.

## 7. Run the Temporal worker (keep this terminal open)

```bash
python worker.py
```

You’ll see logs like "Collection 'document_chunks' loaded".

## 8. In a new terminal, trigger a workflow

```bash
python client.py <file name> <file link>
```

Expected output (example):

```json
{
  "file_id": "file_001",
  "num_chunks": 26,
  "stored": 26,
  "sample": [
    {
      "chunk_index": 0,
      "chunk_text": "Sample PDF Created for testing PDFObject",
      "embedding_preview": [0.0134, -0.0562, 0.0897]
    },
    {
      "chunk_index": 1,
      "chunk_text": "This PDF is three pages long. Three long pages...",
      "embedding_preview": [0.0051, 0.0448, -0.0389]
    }
  ]
}
```

## 9. Verify data in Milvus

```bash
python inspect_milvus.py
```

You should see:

```bash
Connected to Collection: document_chunks
Schema: ...
Total entities in collection: 26

Sample rows:
{'file_id': 'file_001', 'chunk_index': 0, 'chunk_text': '...'}
{'file_id': 'file_001', 'chunk_index': 1, 'chunk_text': '...'}
```

## 10. See the workflow in the UI

Open http://localhost:8080
 → locate your `workflow-...` → Status **Completed**.











