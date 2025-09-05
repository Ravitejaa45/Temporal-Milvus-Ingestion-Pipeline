import sys
import asyncio
import uuid
import json
from temporalio.client import Client
from workflows import DocumentIngestionWorkflow

async def main():
    if len(sys.argv) != 3:
        print("Usage: python client.py <file_id> <file_url>")
        sys.exit(1)

    file_id = sys.argv[1]
    file_url = sys.argv[2]

    for i in range(5):
        try:
            client = await Client.connect("localhost:7233")
            break
        except Exception as e:
            print(f"Retrying Temporal connection in 3s... ({i+1}/5): {e}")
            await asyncio.sleep(3)
    else:
        raise RuntimeError("Failed to connect to Temporal after retries")

    workflow_id = f"workflow-{file_id}-{uuid.uuid4().hex[:6]}"

    handle = await client.start_workflow(
        workflow=DocumentIngestionWorkflow.run,
        args=[file_id, file_url],
        id=workflow_id,
        task_queue="doc-ingest-queue",
    )

    result = await handle.result()
    print("Workflow result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

