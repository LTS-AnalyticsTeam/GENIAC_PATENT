import asyncio
import logging
import signal
import sys

from pipeline import (
    IngestionError,
    JobCancelledError,
    JobManager,
    PipelineConfig,
    JobState,
    run_pipeline,
    run_patent_search_from_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

# Azure SDKのログを抑制（リクエスト/レスポンスヘッダーを非表示）
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)

config = PipelineConfig()
job_manager = JobManager(config.redis_url)

running = True


def handle_exit(signum, frame):
    global running
    logger.info("Received signal %s, shutting down", signum)
    running = False


signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)


async def process_job(job_id: str) -> None:
    payload = job_manager.fetch_payload(job_id)
    if not payload or ("json" not in payload and "text" not in payload):
        logger.error("Job %s payload missing", job_id)
        job_manager.set_state(job_id, JobState(status="failed", detail={"reason": "payload_missing"}))
        return

    if "json" in payload:
        input_bytes = payload["json"].encode("utf-8")
    else:
        input_bytes = payload["text"].encode("utf-8")
    state = job_manager.get_state(job_id)
    if job_manager.is_cancelled(job_id) or (state and state.status == "cancelled"):
        logger.info("Job %s cancelled before processing started", job_id)
        return
    job_manager.set_state(job_id, JobState(status="processing", detail=state.detail if state else {}))

    try:
        # Run the full pipeline (parsing, cosmos query, keyword search, embedding, etc.)
        pipeline_result = await run_pipeline(config, job_manager, job_id, input_bytes)
        logger.info("Job %s: Pipeline completed successfully", job_id)

    except JobCancelledError:
        logger.info("Job %s cancelled during processing", job_id)
    except IngestionError as exc:
        logger.warning("Job %s ingestion failure: %s", job_id, exc)
        state = job_manager.get_state(job_id)
        detail = state.detail if state else {}
        detail = {**(detail or {}), "message": str(exc)}
        job_manager.set_state(job_id, JobState(status="failed", detail=detail))
        job_manager.store_result(job_id, {"error": str(exc), "detail": exc.detail})
    except Exception as exc:  # pragma: no cover
        logger.exception("Job %s unexpected error", job_id)
        state = job_manager.get_state(job_id)
        detail = state.detail if state else {}
        detail = {**(detail or {}), "message": str(exc)}
        job_manager.set_state(job_id, JobState(status="failed", detail=detail))
        job_manager.store_result(job_id, {"error": str(exc)})


async def worker_loop() -> None:
    logger.info("Worker started")
    while running:
        job = job_manager.dequeue_job(timeout=5)
        if not job:
            await asyncio.sleep(1)
            continue
        job_id = job["job_id"]
        if job_manager.is_cancelled(job_id):
            logger.info("Dropping cancelled job %s before processing", job_id)
            continue
        await process_job(job_id)


def main() -> None:
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    finally:
        logger.info("Worker exit")


if __name__ == "__main__":
    main()
