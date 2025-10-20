"""
Cosmos DB Client for Patent Data Retrieval
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from azure.cosmos import CosmosClient, exceptions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CosmosDBClient:
    """Client for interacting with Cosmos DB to retrieve patent data."""

    def __init__(self):
        """Initialize Cosmos DB client with environment variables."""
        self.endpoint = os.getenv("COSMOS_ENDPOINT")
        self.key = os.getenv("COSMOS_KEY")
        self.database_name = os.getenv("COSMOS_DATABASE", "patent_db")
        self.container_name = os.getenv("COSMOS_CONTAINER", "patents")

        if not self.endpoint or not self.key:
            raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY must be set in environment variables")

        # Initialize Cosmos client
        self.client = CosmosClient(self.endpoint, self.key)
        self.database = None
        self.container = None

        self._initialize_database()

    def _initialize_database(self):
        """Initialize database and container references."""
        try:
            self.database = self.client.get_database_client(self.database_name)
            self.container = self.database.get_container_client(self.container_name)
            logger.info(f"Connected to Cosmos DB: {self.database_name}/{self.container_name}")
        except exceptions.CosmosResourceNotFoundError as e:
            logger.error(f"Database or container not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            raise

    def get_all_documents(self, batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Retrieve all documents from Cosmos DB in batches.

        Args:
            batch_size: Number of documents to retrieve per batch

        Yields:
            List of documents in each batch
        """
        query = "SELECT * FROM c"

        try:
            items = self.container.query_items(
                query=query,
                enable_cross_partition_query=True,
                max_item_count=batch_size
            )

            batch = []
            for item in items:
                # Extract relevant fields for processing
                document = self._extract_document_fields(item)
                batch.append(document)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            # Yield remaining documents
            if batch:
                yield batch

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def get_documents_by_date_range(
        self,
        start_date: str,
        end_date: str,
        batch_size: int = 100
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Retrieve documents within a specific date range.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            batch_size: Number of documents per batch

        Yields:
            List of documents in each batch
        """
        query = """
        SELECT * FROM c
        WHERE c.metadata.publication_date >= @start_date
        AND c.metadata.publication_date <= @end_date
        """

        parameters = [
            {"name": "@start_date", "value": start_date},
            {"name": "@end_date", "value": end_date}
        ]

        try:
            items = self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=batch_size
            )

            batch = []
            for item in items:
                document = self._extract_document_fields(item)
                batch.append(document)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

        except Exception as e:
            logger.error(f"Error retrieving documents by date range: {e}")
            raise

    def get_document_by_id(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by patent ID.

        Args:
            patent_id: The patent ID to retrieve

        Returns:
            Document if found, None otherwise
        """
        query = "SELECT * FROM c WHERE c.metadata.patent_id = @patent_id"
        parameters = [{"name": "@patent_id", "value": patent_id}]

        try:
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=1
            ))

            if items:
                return self._extract_document_fields(items[0])
            return None

        except Exception as e:
            logger.error(f"Error retrieving document {patent_id}: {e}")
            raise

    def get_documents_for_incremental_sync(
        self,
        last_sync_timestamp: datetime,
        batch_size: int = 100
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Retrieve documents modified after the last sync timestamp.

        Args:
            last_sync_timestamp: Timestamp of last successful sync
            batch_size: Number of documents per batch

        Yields:
            List of documents in each batch
        """
        # Convert datetime to ISO format string
        timestamp_str = last_sync_timestamp.isoformat()

        query = """
        SELECT * FROM c
        WHERE c.ingest_timestamp > @last_sync
        ORDER BY c.ingest_timestamp ASC
        """

        parameters = [{"name": "@last_sync", "value": timestamp_str}]

        try:
            items = self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=batch_size
            )

            batch = []
            for item in items:
                document = self._extract_document_fields(item)
                batch.append(document)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

        except Exception as e:
            logger.error(f"Error retrieving documents for incremental sync: {e}")
            raise

    @staticmethod
    def _safe_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _normalize_claims(cls, raw_claims: Any) -> Tuple[List[Dict[str, str]], str]:
        claims_list: List[Dict[str, str]] = []

        if not raw_claims:
            return claims_list, ""

        if isinstance(raw_claims, dict):
            iterable = [raw_claims]
        elif isinstance(raw_claims, list):
            iterable = raw_claims
        else:
            iterable = [raw_claims]

        for entry in iterable:
            num = ""
            text = ""

            if isinstance(entry, dict):
                num = cls._safe_text(entry.get("num", "")).strip()
                text = cls._safe_text(entry.get("text", "")).strip()
                if not text:
                    # fallback to alternative keys that sometimes store claim text
                    text = cls._safe_text(entry.get("claim_text", "")).strip()
            elif isinstance(entry, str):
                text = entry.strip()
            else:
                text = cls._safe_text(entry).strip()

            if text:
                claims_list.append({"num": num, "text": text})

        claims_text = "\n".join(claim["text"] for claim in claims_list if claim["text"])
        return claims_list, claims_text

    def _extract_document_fields(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure relevant fields from a Cosmos DB document.

        Args:
            item: Raw document from Cosmos DB

        Returns:
            Structured document with relevant fields
        """
        # Extract metadata
        metadata = item.get("metadata", {})

        # Normalize claims content
        claims_list, claims_text = self._normalize_claims(item.get("claims", []))
        claims_top3_text = "\n".join(
            claim.get("text", "")
            for claim in claims_list[:3]
            if claim.get("text")
        )
        if claims_list:
            number_of_claims = str(len(claims_list))
        else:
            number_of_claims = self._safe_text(item.get("number_of_claims", "0"))

        # Build the document structure for Elasticsearch
        document = {
            # Core identifiers
            "id": item.get("id", ""),
            "patent_id": metadata.get("patent_id", ""),
            "application_number": metadata.get("application_number", ""),

            # Basic information
            "title": metadata.get("title", ""),
            "filing_date": metadata.get("filing_date", ""),
            "publication_date": metadata.get("publication_date", ""),

            # Classification data
            "classification_ipc": [
                ipc.get("code", "") for ipc in metadata.get("classification_ipc", [])
            ],
            "classification_fi": metadata.get("classification_fi", []),
            "f_term": metadata.get("f_term", []),
            "theme_code": metadata.get("theme_code", []),

            # Text content for embedding
            "summary": item.get("summary", ""),

            # Keywords and topics
            "keywords": metadata.get("keywords", []),
            "topics": metadata.get("topics", []),

            # Reference information
            "reference": metadata.get("reference", {}),

            # Additional fields
            "priority_date": item.get("priority_date", ""),
            "number_of_claims": number_of_claims,
            "claims": claims_list,
            "claims_text": claims_text,
            "claims_top3_text": claims_top3_text,
            "applicants": item.get("applicants", []),
            "inventors": item.get("inventors", []),
            "country": item.get("country", "JP"),
            "kind_code": item.get("kind_code", ""),
            "language": item.get("language", "ja"),

            # System fields
            "source_file": item.get("source_file", ""),
            "ingest_timestamp": item.get("ingest_timestamp", ""),
            "parse_version": item.get("parse_version", ""),
            "file_size": item.get("file_size", 0),
            "checksum": item.get("checksum", "")
        }

        return document

    def get_total_document_count(self) -> int:
        """
        Get the total count of documents in the container.

        Returns:
            Total number of documents
        """
        query = "SELECT VALUE COUNT(1) FROM c"

        try:
            result = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            raise

    def close(self):
        """Close the Cosmos DB connection."""
        # CosmosClient doesn't require explicit closing
        logger.info("Cosmos DB client closed")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize client
    client = CosmosDBClient()

    # Get total count
    total = client.get_total_document_count()
    print(f"Total documents: {total}")

    # Get first batch of documents
    for batch in client.get_all_documents(batch_size=10):
        print(f"Retrieved {len(batch)} documents")
        if batch:
            print(f"First document patent_id: {batch[0].get('patent_id')}")
        break

    client.close()
