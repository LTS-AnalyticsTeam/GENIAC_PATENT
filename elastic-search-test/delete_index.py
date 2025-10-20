#!/usr/bin/env python3
"""
Elasticsearch Index Deletion Script
This script deletes the current patent_vectors index from Elasticsearch.
"""
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_elasticsearch_index():
    """Delete the Elasticsearch index."""

    # Get configuration from environment
    es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    index_name = os.getenv("ELASTICSEARCH_INDEX", "patent_vectors")
    request_timeout = int(os.getenv("ELASTICSEARCH_REQUEST_TIMEOUT", "30"))

    logger.info(f"Connecting to Elasticsearch at: {es_host}")
    logger.info(f"Target index for deletion: {index_name}")

    try:
        # Initialize Elasticsearch client
        es = Elasticsearch(
            es_host,
            request_timeout=request_timeout
        )

        # Check connection
        if not es.ping():
            logger.error(f"Cannot connect to Elasticsearch at {es_host}")
            return False

        logger.info("Successfully connected to Elasticsearch")

        # Check if index exists
        if not es.indices.exists(index=index_name):
            logger.warning("Index '%s' does not exist", index_name)
            return True

        # Get index stats before deletion
        try:
            count_response = es.count(index=index_name)
            document_count = count_response["count"]
            logger.info(f"Index '{index_name}' contains {document_count} documents")
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            document_count = "unknown"

        # Confirm deletion
        print(f"\n⚠️  WARNING: You are about to delete the Elasticsearch index '{index_name}'")
        print(f"   This index contains {document_count} documents")
        print(f"   This action cannot be undone!")

        confirmation = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()

        if confirmation not in ['yes', 'y']:
            logger.info("Index deletion cancelled by user")
            return False

        # Delete the index
        logger.info(f"Deleting index '{index_name}'...")
        delete_response = es.indices.delete(index=index_name)

        if delete_response.get("acknowledged"):
            logger.info(f"✅ Successfully deleted index '{index_name}'")
            logger.info(f"   Deleted {document_count} documents")
            logger.info(f"   Deletion completed at: {datetime.now().isoformat()}")
            return True
        else:
            logger.error(f"❌ Failed to delete index '{index_name}'")
            return False

    except Exception as e:
        logger.error(f"Error during index deletion: {e}")
        return False

    finally:
        try:
            es.close()
        except Exception:
            pass


def main():
    """Main function."""
    print("Elasticsearch Index Deletion Tool")
    print("=" * 40)

    success = delete_elasticsearch_index()

    if success:
        print("\n✅ Index deletion completed successfully!")
        print("You can now recreate the index with fresh data.")
        sys.exit(0)
    else:
        print("\n❌ Index deletion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
