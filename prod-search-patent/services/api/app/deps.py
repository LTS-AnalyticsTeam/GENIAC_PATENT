from __future__ import annotations

from fastapi import Depends

from pipeline import JobManager, PipelineConfig

from .settings import get_settings


def get_pipeline_config():
    return PipelineConfig()


def get_job_manager(config: PipelineConfig = Depends(get_pipeline_config)) -> JobManager:
    return JobManager(config.redis_url)

