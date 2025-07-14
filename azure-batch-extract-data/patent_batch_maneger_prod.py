#!/usr/bin/env python3
"""
Azure Batch – 公式ドキュメントどおりの最短パターン
20.04 コンテナイメージ (microsoft-azure-batch/ubuntu-server-container)
ノード 4 台 (D4s_v3) で tar.gz 18 本を並列処理
"""

import os
import time
from datetime import timedelta
from urllib.parse import quote

from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.batch.models import (
    ComputeNodeState,
    ContainerConfiguration,
    ContainerRegistry,
    EnvironmentSetting,
    ImageReference,
    JobAddParameter,
    JobState,
    OnAllTasksComplete,
    OnTaskFailure,
    PoolAddParameter,
    PoolInformation,
    PoolState,
    ResourceFile,
    TaskAddParameter,
    TaskConstraints,
    TaskContainerSettings,
    TaskState,
    VirtualMachineConfiguration,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── 認証情報（環境変数から取得） ─────────
BATCH_ACCOUNT_NAME = os.getenv('BATCH_ACCOUNT_NAME')
BATCH_ACCOUNT_URL = os.getenv('BATCH_ACCOUNT_URL')
BATCH_ACCOUNT_KEY = os.getenv('BATCH_ACCOUNT_KEY')

ACR_SERVER = os.getenv('ACR_SERVER')
ACR_USER = os.getenv('ACR_USER')
ACR_PASSWORD = os.getenv('ACR_PASSWORD')

AZURE_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
AZURE_TENANT_ID = os.getenv('AZURE_TENANT_ID')
AZURE_CLIENT_SECRET = os.getenv('AZURE_CLIENT_SECRET')

STORAGE_ACCOUNT = os.getenv('STORAGE_ACCOUNT')
CONTAINER_NAME = os.getenv('CONTAINER_NAME', 'patent-all')
CONTAINER_SAS = os.getenv('CONTAINER_SAS')

DOCKER_IMAGE = os.getenv('DOCKER_IMAGE')
POOL_ID = os.getenv('POOL_ID', 'jp-patent-extract-pool-prod')
VM_SIZE = os.getenv('VM_SIZE', 'Standard_D8s_v3')
NUM_ARCHIVES = int(os.getenv('NUM_ARCHIVES', '18'))

# Validate required environment variables
required_vars = [
    'BATCH_ACCOUNT_NAME', 'BATCH_ACCOUNT_URL', 'BATCH_ACCOUNT_KEY',
    'ACR_SERVER', 'ACR_USER', 'ACR_PASSWORD',
    'AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_SECRET',
    'STORAGE_ACCOUNT', 'CONTAINER_SAS', 'DOCKER_IMAGE'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
# ────────────────────────────────────────────────


class BatchMgr:
    def __init__(self):
        self.cli = BatchServiceClient(
            SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY),
            BATCH_ACCOUNT_URL)
        print("[Batch] authenticated")

    # ---------- プール ----------
    def create_pool(self):
        try:
            if self.cli.pool.get(POOL_ID):
                print("[Pool] exists")
                return
        except Exception:
            pass

        vm_conf = VirtualMachineConfiguration(
            image_reference=ImageReference(
                publisher="microsoft-azure-batch",
                offer="ubuntu-server-container",
                sku="20-04-lts",
                version="latest",
            ),
            container_configuration=ContainerConfiguration(
                type="dockerCompatible",
                container_image_names=[DOCKER_IMAGE],            # ← プリフェッチ
                container_registries=[ContainerRegistry(
                    registry_server=ACR_SERVER,
                    user_name=ACR_USER,
                    password=ACR_PASSWORD)]
            ),
            node_agent_sku_id="batch.node.ubuntu 20.04",
        )

        pool = PoolAddParameter(
            id=POOL_ID, vm_size=VM_SIZE, virtual_machine_configuration=vm_conf,
            target_low_priority_nodes=6, task_slots_per_node=1)
        self.cli.pool.add(pool)
        print("[Pool] created")

    def wait_pool(self):
        while True:
            p = self.cli.pool.get(POOL_ID)
            ready = p.state == PoolState.active and p.current_low_priority_nodes
            idle = [n for n in self.cli.compute_node.list(POOL_ID)
                    if n.state == ComputeNodeState.idle]
            if ready and idle:
                print(f"[Pool] ready – idle={len(idle)}")
                return
            print("[Pool] waiting…")
            time.sleep(30)

    # ---------- ジョブ & タスク ----------
    def create_job(self, job_id):
        try:
            if self.cli.job.get(job_id).state not in {JobState.completed, JobState.terminating}:
                print("[Job] exists")
                return
            self.cli.job.delete(job_id)
        except Exception:
            pass
        self.cli.job.add(JobAddParameter(
            id=job_id, pool_info=PoolInformation(pool_id=POOL_ID),
            on_all_tasks_complete=OnAllTasksComplete.no_action,
            on_task_failure=OnTaskFailure.no_action))
        print("[Job] created")

    def add_task(self, job_id, blob_path, idx):
        name = os.path.basename(blob_path)
        sas = (f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/"
               f"{quote(blob_path, safe='/')}?{CONTAINER_SAS}")
        self.cli.task.add(job_id, TaskAddParameter(
            id=f"task_{idx:03d}",
            resource_files=[ResourceFile(http_url=sas, file_path=name)],
            # ENTRYPOINT が /app/extract_data_processor.py を呼ぶ想定
            command_line="python /app/extract_data_processor.py",
            container_settings=TaskContainerSettings(
                image_name=DOCKER_IMAGE,
                # Batch が強制する WORKDIR を上書きして /app に戻す
                container_run_options="--rm --workdir /app"
            ),
            environment_settings=[
                EnvironmentSetting(name="ARCHIVE_PATH",        value=name),
                EnvironmentSetting(name="GZ_FILE_BLOB",        value=name),
                EnvironmentSetting(name="AZURE_CLIENT_ID",     value=AZURE_CLIENT_ID),
                EnvironmentSetting(name="AZURE_TENANT_ID",     value=AZURE_TENANT_ID),
                EnvironmentSetting(name="AZURE_CLIENT_SECRET", value=AZURE_CLIENT_SECRET),
            ],
            constraints=TaskConstraints(
                max_wall_clock_time=timedelta(hours=8), max_task_retry_count=1)
        ))

    def monitor(self, job_id):
        while True:
            t = list(self.cli.task.list(job_id))
            done = [x for x in t if x.state == TaskState.completed]
            fail = [x for x in done if x.execution_info.exit_code]
            print(f"[Job] {len(done)}/{len(t)} done (fail={len(fail)})")
            if len(done) == len(t):
                if fail:
                    raise RuntimeError(f"{len(fail)} task(s) failed")
                print("[Job] ALL SUCCESS")
                return
            time.sleep(60)


def main():
    mgr = BatchMgr()
    mgr.create_pool()
    mgr.wait_pool()
    job = "jp-extract-job-prod"
    mgr.create_job(job)
    for i in range(1, NUM_ARCHIVES+1):
        mgr.add_task(job, f"02.国内特許文献データ/result_{i}.tar.gz", i)
    mgr.monitor(job)


if __name__ == "__main__":
    main()
