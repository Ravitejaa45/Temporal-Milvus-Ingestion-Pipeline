import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.worker import Worker
from temporalio.client import Client

from workflows import DocumentIngestionWorkflow
import activities

async def main():
    client = await Client.connect("localhost:7233")

    activity_executor = ThreadPoolExecutor()
    worker = Worker(
        client,
        task_queue="doc-ingest-queue",
        workflows=[DocumentIngestionWorkflow],
        activities=[
            activities.fetch_document,
            activities.parse_document,
            activities.generate_embeddings,
            activities.store_in_milvus,
        ],
        activity_executor=activity_executor,
    )
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
