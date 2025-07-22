import os
import json
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# .envファイルから環境変数を読み込む
load_dotenv()

COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

MAX_DOC_SIZE = 2 * 1024 * 1024  # 2MB

client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
db = client.create_database_if_not_exists(DATABASE_NAME)
container = db.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/metadata/patent_id"),
    offer_throughput=400
)

output_dir = "output_files"
skipped_files = []
for filename in os.listdir(output_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(output_dir, filename)
        # まずファイルサイズでスキップ
        # if os.path.getsize(file_path) > MAX_DOC_SIZE:
        #     print(f"Skipped (too large): {filename}")
        #     skipped_files.append(filename)
        #     continue
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            if "id" not in doc:
                patent_id = doc.get("metadata", {}).get("patent_id", "")
                publication_date = doc.get("metadata", {}).get("publication_date", "")
                doc["id"] = f"{patent_id}_{publication_date}"
            # 文字列化したときのバイト数もチェック
            doc_bytes = json.dumps(doc, ensure_ascii=False).encode("utf-8")
            if len(doc_bytes) > MAX_DOC_SIZE:
                print(f"Skipped (JSON too large): {filename}")
                skipped_files.append(filename)
                continue
            try:
                container.upsert_item(doc)
                print(f"Uploaded: {filename}")
            except exceptions.CosmosHttpResponseError as e:
                if hasattr(e, 'status_code') and e.status_code == 413:
                    print(f"Skipped (CosmosDB 413): {filename}")
                    skipped_files.append(filename)
                else:
                    print(f"Error uploading {filename}: {e}")

if skipped_files:
    print("\n=== Skipped files (not uploaded) ===")
    for fname in skipped_files:
        print(fname)
else:
    print("\nAll files uploaded (no oversized files).")
