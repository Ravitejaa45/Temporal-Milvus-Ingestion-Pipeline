from temporalio import workflow
from temporalio.common import RetryPolicy
from datetime import timedelta

@workflow.defn
class DocumentIngestionWorkflow:
    @workflow.run
    async def run(self, file_id: str, file_url: str) -> dict:
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
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        chunk_embeddings = await workflow.execute_activity(
            "generate_embeddings",
            args=[chunks],
            schedule_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )

        await workflow.execute_activity(
            "store_in_milvus",
            args=[chunk_embeddings],
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        return {
            "file_id": file_id,
            "num_chunks": len(chunk_embeddings),
            "sample": chunk_embeddings[:2],
        }
