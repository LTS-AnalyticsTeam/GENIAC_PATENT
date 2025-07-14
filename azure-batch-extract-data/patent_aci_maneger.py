import os
import time

from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    Container,
    ContainerGroup,
    ContainerGroupRestartPolicy,
    EnvironmentVariable,
    ImageRegistryCredential,
    OperatingSystemTypes,
    ResourceRequests,
    ResourceRequirements,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ACIParallelRunner:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = os.getenv('AZURE_RESOURCE_GROUP')
        self.location = os.getenv('AZURE_LOCATION', 'japaneast')

        # Validate required environment variables
        required_vars = ['AZURE_SUBSCRIPTION_ID', 'AZURE_RESOURCE_GROUP']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.aci_client = ContainerInstanceManagementClient(
            self.credential,
            self.subscription_id
        )
        print("ACI client initialized")

    def create_container_group(self, container_name, tar_gz_file):
        """コンテナインスタンスを作成"""

        # 環境変数
        environment_variables = [
            EnvironmentVariable(name="GZ_FILE_BLOB", value=tar_gz_file),
            EnvironmentVariable(name="AZURE_CLIENT_ID", value=os.getenv('AZURE_CLIENT_ID')),
            EnvironmentVariable(name="AZURE_TENANT_ID", value=os.getenv('AZURE_TENANT_ID')),
            EnvironmentVariable(name="AZURE_CLIENT_SECRET", value=os.getenv('AZURE_CLIENT_SECRET'))
        ]

        # コンテナ定義
        container = Container(
            name=container_name,
            image=os.getenv('DOCKER_IMAGE'),
            resources=ResourceRequirements(
                requests=ResourceRequests(
                    memory_in_gb=4,
                    cpu=2
                )
            ),
            environment_variables=environment_variables,
            command=["python", "extract_data_processor.py"]
        )

        # コンテナグループ定義
        container_group = ContainerGroup(
            location=self.location,
            containers=[container],
            os_type=OperatingSystemTypes.linux,  # Linux に戻す
            restart_policy=ContainerGroupRestartPolicy.never,
            image_registry_credentials=[
                ImageRegistryCredential(
                    server=os.getenv('ACR_SERVER'),
                    username=os.getenv('ACR_USER'),
                    password=os.getenv('ACR_PASSWORD')
                )
            ]
        )

        # 作成実行
        try:
            result = self.aci_client.container_groups.begin_create_or_update(
                self.resource_group,
                container_name,
                container_group
            )
            print(f"Creating container: {container_name} for {tar_gz_file}")
            return result
        except Exception as e:
            print(f"Error creating container {container_name}: {e}")
            return None

    def wait_for_completion(self, container_names):
        """全コンテナの完了を待機"""
        while True:
            completed = 0
            failed = 0

            for container_name in container_names:
                try:
                    container_group = self.aci_client.container_groups.get(
                        self.resource_group,
                        container_name
                    )

                    if container_group.instance_view:
                        state = container_group.instance_view.state
                        if state == "Succeeded":
                            completed += 1
                        elif state == "Failed":
                            failed += 1
                            print(f"Container {container_name} failed")
                except Exception:
                    # コンテナが見つからない場合はスキップ
                    pass

            print(f"Progress: {completed}/{len(container_names)} completed, {failed} failed")

            if completed + failed == len(container_names):
                print("All containers finished!")
                break

            time.sleep(30)  # 30秒待機

    def cleanup_containers(self, container_names):
        """完了したコンテナを削除"""
        for container_name in container_names:
            try:
                self.aci_client.container_groups.begin_delete(
                    self.resource_group,
                    container_name
                )
                print(f"Deleted container: {container_name}")
            except Exception as e:
                print(f"Error deleting {container_name}: {e}")


def main():
    runner = ACIParallelRunner()

    # 18個のtar.gzファイル
    tar_gz_files = [f'result_{i}.tar.gz' for i in range(1, 19)]

    # コンテナ名リスト
    container_names = [f"jp-extract-{i:02d}" for i in range(1, 19)]

    BATCH_SIZE = 5
    for idx in range(0, len(tar_gz_files), BATCH_SIZE):
        batch_names = container_names[idx:idx + BATCH_SIZE]
        batch_files = tar_gz_files[idx:idx + BATCH_SIZE]

        # バッチ起動
        for name, blob in zip(batch_names, batch_files):
            runner.create_container_group(name, blob)
            time.sleep(2)

        # 完了待ち
        runner.wait_for_completion(batch_names)

        # クリーンアップ
        runner.cleanup_containers(batch_names)

    print("ACI parallel processing finished!")


if __name__ == "__main__":
    main()
