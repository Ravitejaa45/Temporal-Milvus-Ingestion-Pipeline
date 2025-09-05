from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from datetime import timedelta
import asyncio
import os
from urllib.parse import urlparse

ALLOWED_EXTS = {".docx", ".doc", ".pdf", ".xlsx", ".xls"}
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "128"))
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))
UPSERT_CONCURRENCY = int(os.getenv("UPSERT_CONCURRENCY", "8"))

def _ext_from_url(url: str) -> str:
    path = urlparse(url).path
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""

def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]

def _preview_vec(vec, k=3):
    return list(vec[:k]) if vec else []

@workflow.defn
class DocumentIngestionWorkflow:
    @workflow.run
    async def run(self, file_id: str, file_url: str) -> dict:
        ext = _ext_from_url(file_url)
        if ext and ext not in ALLOWED_EXTS:
            raise ApplicationError(
                f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
                non_retryable=True,
            )

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=5,
        )

        path = await workflow.execute_activity(
            "fetch_document",
            args=[file_url, file_id],
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        chunks = await workflow.execute_activity(
            "parse_document",
            args=[path],
            schedule_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )
        if not chunks:
            return {"file_id": file_id, "num_chunks": 0, "stored": 0, "sample": []}

        records = [
            {"file_id": file_id, "chunk_index": i, "chunk_text": txt}
            for i, txt in enumerate(chunks)
        ]

        embed_batches = list(_chunk(records, EMBED_BATCH_SIZE))
        embed_results_all = []
        for i in range(0, len(embed_batches), EMBED_CONCURRENCY):
            window = embed_batches[i : i + EMBED_CONCURRENCY]
            handles = [
                workflow.start_activity(
                    "generate_embeddings",
                    args=[[r["chunk_text"] for r in batch]],
                    schedule_to_close_timeout=timedelta(seconds=180),
                    retry_policy=retry_policy,
                )
                for batch in window
            ]
            batch_vectors_lists = await asyncio.gather(*handles)
            embed_results_all.extend(batch_vectors_lists)

        out_records = []
        for batch, vectors in zip(embed_batches, embed_results_all):
            if len(vectors) != len(batch):
                raise ApplicationError("Embedding size mismatch within a batch", non_retryable=True)
            for rec, vec in zip(batch, vectors):
                out_records.append({**rec, "embedding": vec})

        sample = [
            {
                "chunk_index": r["chunk_index"],
                "chunk_text": (r["chunk_text"][:200] + ("…" if len(r["chunk_text"]) > 200 else "")),
                "embedding_preview": _preview_vec(r["embedding"], 3),
            }
            for r in out_records[:2]
        ]

        upsert_batches = list(_chunk(out_records, UPSERT_BATCH_SIZE))
        stored_total = 0
        for i in range(0, len(upsert_batches), UPSERT_CONCURRENCY):
            window = upsert_batches[i : i + UPSERT_CONCURRENCY]
            handles = [
                workflow.start_activity(
                    "store_in_milvus",
                    args=[batch],
                    schedule_to_close_timeout=timedelta(seconds=180),
                    retry_policy=retry_policy,
                )
                for batch in window
            ]
            results = await asyncio.gather(*handles)
            stored_total += sum(r.get("inserted", 0) for r in results)

        return {
            "file_id": file_id,
            "num_chunks": len(records),
            "stored": stored_total,
            "sample": sample,
        }
