import io
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SimpleJPProcessor:
    def __init__(self):
        # 環境変数から認証情報を取得
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        # Service Principal認証を明示的に使用
        if client_id and client_secret and tenant_id:
            print("Using Service Principal authentication")
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
        else:
            print("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()

        # 環境変数からストレージアカウントURLを取得
        storage_account_url = os.getenv('STORAGE_ACCOUNT_URL')
        if not storage_account_url:
            raise ValueError("STORAGE_ACCOUNT_URL environment variable is required")

        self.blob_client = BlobServiceClient(account_url=storage_account_url, credential=credential)
        self.target_jp_set = set()

    def load_jp_numbers(self):
        """4万件のJP番号をHashSetに読み込み"""
        # 環境変数から設定を取得
        container = os.getenv('CONTAINER_NAME', 'patent-all')
        csv_blob = "CSV1.csv"

        blob_client = self.blob_client.get_blob_client(container=container, blob=csv_blob)
        csv_content = blob_client.download_blob().content_as_text()
        df = pd.read_csv(io.StringIO(csv_content))

        # syutugan列（出願番号）を対象とし、重複を除去
        if 'syutugan' in df.columns:
            # 重複除去してsetに変換
            unique_jp_numbers = df['syutugan'].dropna().drop_duplicates().astype(str)
            self.target_jp_set = set(unique_jp_numbers)
            print(f"Loaded {len(self.target_jp_set)} unique target JP numbers from 'syutugan' column")
            print(f"Sample JP numbers: {list(self.target_jp_set)[:5]}")  # 最初の5件を表示
        else:
            print(f"Error: 'syutugan' column not found. Available columns: {df.columns.tolist()}")
            self.target_jp_set = set()

    def download_tar_gz_file(self, blob_name, local_path):
        """tar.gzファイルをダウンロード"""
        container = os.getenv('CONTAINER_NAME', 'patent-all')

        # まず該当ファイルを検索
        container_client = self.blob_client.get_container_client(container)
        blob_list = container_client.list_blobs(name_starts_with="02")

        target_file = None
        for blob in blob_list:
            if blob_name in blob.name:
                target_file = blob.name
                print(f"Found target file: {target_file}")
                break

        if not target_file:
            raise FileNotFoundError(f"Could not find blob containing: {blob_name}")

        # 見つかったファイルをダウンロード
        blob_client = self.blob_client.get_blob_client(container=container, blob=target_file)
        with open(local_path, 'wb') as f:
            f.write(blob_client.download_blob().readall())
        print(f"Downloaded: {target_file}")

    def extract_tar_gz(self, tar_gz_path, extract_to):
        """tar.gzファイルを展開"""
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
            print(f"Extracted tar.gz to {extract_to}")

    def find_text_files(self, extract_dir):
        """展開されたディレクトリからtext.txtファイルを再帰的に検索"""
        text_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file == 'text.txt':
                    file_path = os.path.join(root, file)
                    # JP番号をディレクトリ名から取得
                    jp_number = os.path.basename(os.path.dirname(file_path))
                    text_files.append((jp_number, file_path))

        print(f"Found {len(text_files)} text.txt files")
        return text_files

    def process_text_files(self, text_files):
        """text.txtファイルからJP番号に一致するファイルを特定"""
        matched_files = {}

        for jp_number, file_path in text_files:
            # JP番号が対象リストに含まれているかチェック
            if jp_number in self.target_jp_set:
                matched_files[jp_number] = file_path
                print(f"Found target JP: {jp_number}")

        print(f"Total matched {len(matched_files)} JP patents")
        return matched_files

    def upload_results(self, matched_files):
        """マッチしたtext.txtファイルをJP番号.txtとしてアップロード"""
        container = "patent-extract-csv1"  # 新しいコンテナ

        def upload_single(jp_number, file_path):
            # JP番号.txtとしてアップロード
            blob_name = f"{jp_number}.txt"
            blob_client = self.blob_client.get_blob_client(container=container, blob=blob_name)

            # ファイルの内容を読み取ってアップロード
            with open(file_path, 'rb') as f:
                blob_client.upload_blob(f.read(), overwrite=True)
            print(f"Uploaded: {blob_name}")

        # 並列アップロード
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_single, jp, path) for jp, path in matched_files.items()]
            for future in futures:
                future.result()

        print(f"Uploaded {len(matched_files)} files to container: patent-extract-csv1")


def main():
    processor = SimpleJPProcessor()

    # ハードコーディング：処理するtar.gzファイル名
    # 環境変数から取得（Batchタスクで設定される）
    tar_gz_file = os.getenv('GZ_FILE_BLOB', 'result_1.tar.gz')  # デフォルト値設定
    pre_downloaded = os.path.exists(tar_gz_file)

    print(f"Processing file: {tar_gz_file}")

    # JP番号リストを読み込み
    processor.load_jp_numbers()

    # 一時ディレクトリで処理
    with tempfile.TemporaryDirectory() as temp_dir:
        local_tar_gz_path = os.path.join(temp_dir, tar_gz_file)
        extract_dir = os.path.join(temp_dir, 'extracted')

        # ダウンロード
        if pre_downloaded:
            print(f"Use pre-downloaded archive {tar_gz_file}")
            shutil.copy2(tar_gz_file, local_tar_gz_path)
        else:
            print(f"Downloading {tar_gz_file} from Blob…")
            processor.download_tar_gz_file(tar_gz_file, local_tar_gz_path)

        # tar.gz展開
        print(f"Extracting {tar_gz_file}...")
        processor.extract_tar_gz(local_tar_gz_path, extract_dir)

        # text.txtファイルを検索
        print("Finding text.txt files...")
        text_files = processor.find_text_files(extract_dir)

        # 処理
        print("Processing text files...")
        matched_files = processor.process_text_files(text_files)

        # アップロード
        if matched_files:
            processor.upload_results(matched_files)
        else:
            print("No matching JP numbers found in this file")

        print("Complete!")


if __name__ == "__main__":
    main()
