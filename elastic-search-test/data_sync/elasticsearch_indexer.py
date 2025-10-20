"""
Elasticsearch Indexer for Patent Vector Data
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

logger = logging.getLogger(__name__)


class ElasticsearchIndexer:
    """Indexer for storing patent documents with vectors in Elasticsearch."""

    def __init__(self):
        """Initialize Elasticsearch client and configuration."""
        self.es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "patent_vectors")
        self.batch_size = int(os.getenv("ELASTICSEARCH_BATCH_SIZE", "100"))
        self.request_timeout = int(os.getenv("ELASTICSEARCH_REQUEST_TIMEOUT", "120"))  # Increased from 30 to 120 seconds
        self.index_creation_timeout = int(os.getenv("ELASTICSEARCH_INDEX_CREATION_TIMEOUT", "300"))  # 5 minutes for index creation
        self.summary_vector_field = os.getenv("ES_SUMMARY_VECTOR_FIELD", "summary_vector")
        self.claims1_vector_field = os.getenv("ES_CLAIMS1_VECTOR_FIELD", "claims1_vector")
        legacy_vector_field = os.getenv("ES_VECTOR_FIELD")
        # Preserve legacy attribute for compatibility, defaulting to summary vector.
        self.vector_field = legacy_vector_field or self.summary_vector_field
        self.vector_dims = int(os.getenv("ES_VECTOR_DIMS", "3072"))
        self.claims_top3_field = os.getenv("ES_CLAIMS_TOP3_FIELD", "claims_top3_text")
        self.summary_vector_weight = float(os.getenv("ES_SUMMARY_VECTOR_WEIGHT", "0.6"))
        self.claims_vector_weight = float(os.getenv("ES_CLAIMS_VECTOR_WEIGHT", "0.4"))
        self.min_vector_score = float(os.getenv("ES_MIN_VECTOR_SCORE", "0.4"))
        self.max_num_candidates = int(os.getenv("ES_MAX_NUM_CANDIDATES", "10000"))

        # Initialize Elasticsearch client
        self.es = Elasticsearch(
            self.es_host,
            request_timeout=self.request_timeout
        )

        # Check connection
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {self.es_host}")

        logger.info(f"Connected to Elasticsearch at {self.es_host}")

        # Statistics
        self.total_indexed = 0
        self.total_failed = 0
        self.failed_documents = []

    def create_index(self, force_recreate: bool = False):
        """
        Create the Elasticsearch index with appropriate mappings.

        Args:
            force_recreate: If True, delete and recreate the index
        """
        # Check if index exists
        if self.es.indices.exists(index=self.index_name):
            if force_recreate:
                logger.warning(f"Deleting existing index: {self.index_name}")
                self.es.indices.delete(index=self.index_name)
            else:
                logger.info(f"Index {self.index_name} already exists")
                return

        # Define index mappings
        mappings = {
            "mappings": {
                "properties": {
                    # Core identifiers
                    "id": {"type": "keyword"},
                    "patent_id": {"type": "keyword"},
                    "application_number": {"type": "keyword"},

                    # Basic information
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "filing_date": {"type": "date", "format": "yyyyMMdd||strict_date_optional_time"},
                    "publication_date": {"type": "date", "format": "yyyyMMdd||strict_date_optional_time"},
                    "priority_date": {"type": "date", "format": "yyyyMMdd||strict_date_optional_time||epoch_millis"},

                    # Classification data
                    "classification_ipc": {"type": "keyword"},
                    "classification_fi": {"type": "keyword"},
                    "f_term": {"type": "keyword"},
                    "theme_code": {"type": "keyword"},

                    # Text content
                    "summary": {
                        "type": "text"
                    },
                    "claims": {
                        "type": "object",
                        "properties": {
                            "num": {"type": "keyword"},
                            "text": {"type": "text"}
                        }
                    },
                    "claims_text": {
                        "type": "text"
                    },
                    self.claims_top3_field: {
                        "type": "text"
                    },

                    # Vector fields for similarity search
                    self.summary_vector_field: {
                        "type": "dense_vector",
                        "dims": self.vector_dims,
                        "index": True,
                        "similarity": "cosine"
                    },
                    self.claims1_vector_field: {
                        "type": "dense_vector",
                        "dims": self.vector_dims,
                        "index": True,
                        "similarity": "cosine"
                    },

                    # Reference information
                    "reference": {
                        "type": "object",
                        "properties": {
                            "exist": {"type": "boolean"},
                            "syutugan": {"type": "keyword"},
                            "himotsuki": {"type": "keyword"}
                        }
                    },

                    # Additional fields
                    "applicants": {"type": "keyword"},
                    "inventors": {"type": "keyword"},
                    "country": {"type": "keyword"},
                    "kind_code": {"type": "keyword"},
                    "language": {"type": "keyword"},

                    # System fields
                    "source_file": {"type": "keyword"},
                    "ingest_timestamp": {"type": "date"},
                    "parse_version": {"type": "keyword"},
                    "file_size": {"type": "long"},
                    "checksum": {"type": "keyword"},

                    # Embedding metadata
                    "embedding_model": {"type": "keyword"},
                    "embedding_generated": {"type": "boolean"},
                    "indexed_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        }

        # Create index with extended timeout
        try:
            logger.info(f"Creating index {self.index_name} with extended timeout ({self.index_creation_timeout}s)...")

            # Create a separate client with longer timeout for index creation
            index_creation_client = Elasticsearch(
                self.es_host,
                request_timeout=self.index_creation_timeout
            )

            index_creation_client.indices.create(index=self.index_name, body=mappings)
            index_creation_client.close()

            logger.info(f"Successfully created index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            raise

    def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document.

        Args:
            document: Document to index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add indexing timestamp
            document["indexed_at"] = datetime.utcnow().isoformat()

            # Use patent_id as document ID
            doc_id = document.get("patent_id", document.get("id"))

            # Index the document
            response = self.es.index(
                index=self.index_name,
                id=doc_id,
                body=document
            )

            if response.get("result") in ["created", "updated"]:
                self.total_indexed += 1
                return True
            else:
                self.total_failed += 1
                self.failed_documents.append(doc_id)
                return False

        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            self.total_failed += 1
            self.failed_documents.append(doc_id)
            return False

    def document_exists(self, patent_id: str) -> bool:
        """
        Check if a document already exists in the index.

        Args:
            patent_id: Patent ID to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            return self.es.exists(index=self.index_name, id=patent_id)
        except Exception as e:
            logger.error(f"Error checking document existence {patent_id}: {e}")
            return False

    def bulk_check_existence(self, patent_ids: List[str]) -> Dict[str, bool]:
        """
        Check existence of multiple documents efficiently.

        Args:
            patent_ids: List of patent IDs to check

        Returns:
            Dictionary mapping patent_id to existence status
        """
        if not patent_ids:
            return {}

        try:
            # Use multi-get to check existence efficiently
            body = {"ids": patent_ids}
            response = self.es.mget(
                index=self.index_name,
                body=body,
                _source=False  # We only need to know if they exist
            )

            existence_map = {}
            for doc in response["docs"]:
                patent_id = doc["_id"]
                existence_map[patent_id] = doc.get("found", False)

            return existence_map

        except Exception as e:
            logger.error(f"Error checking bulk document existence: {e}")
            # Return all as non-existing to be safe
            return {pid: False for pid in patent_ids}

    def bulk_index_documents(self, documents: List[Dict[str, Any]], skip_existing: bool = True) -> Dict[str, Any]:
        """
        Bulk index multiple documents with duplicate prevention.

        Args:
            documents: List of documents to index
            skip_existing: If True, skip documents that already exist

        Returns:
            Dictionary with indexing results
        """
        if not documents:
            return {"indexed": 0, "failed": 0, "errors": [], "skipped": 0}

        # Filter out documents without embeddings
        valid_documents = []
        skipped_no_embedding = 0

        for doc in documents:
            has_summary_vec = bool(doc.get(self.summary_vector_field))
            has_claims_vec = bool(doc.get(self.claims1_vector_field))
            if not (has_summary_vec or has_claims_vec):
                logger.warning(
                    "Skipping document without embeddings (summary/claims1 missing): %s",
                    doc.get("patent_id"),
                )
                skipped_no_embedding += 1
                continue
            valid_documents.append(doc)

        if not valid_documents:
            return {"indexed": 0, "failed": 0, "errors": [], "skipped": skipped_no_embedding}

        # Check for existing documents if skip_existing is True
        skipped_existing = 0
        documents_to_index = valid_documents

        if skip_existing:
            patent_ids = [doc.get("patent_id", doc.get("id")) for doc in valid_documents]
            existence_map = self.bulk_check_existence(patent_ids)

            documents_to_index = []
            for doc in valid_documents:
                patent_id = doc.get("patent_id", doc.get("id"))
                if existence_map.get(patent_id, False):
                    logger.debug(f"Skipping existing document: {patent_id}")
                    skipped_existing += 1
                else:
                    documents_to_index.append(doc)

        if not documents_to_index:
            logger.info(f"All documents already exist. Skipped: {skipped_existing}")
            return {"indexed": 0, "failed": 0, "errors": [], "skipped": skipped_existing + skipped_no_embedding}

        # Prepare bulk actions
        actions = []
        for doc in documents_to_index:
            # Add indexing timestamp
            doc["indexed_at"] = datetime.utcnow().isoformat()

            # Prepare action - use create instead of index to prevent overwrites
            action = {
                "_op_type": "create",  # This will fail if document already exists
                "_index": self.index_name,
                "_id": doc.get("patent_id", doc.get("id")),
                "_source": doc
            }
            actions.append(action)

        # Perform bulk indexing
        try:
            success_count = 0
            failed_count = 0
            errors = []

            # Process in smaller chunks to handle errors better
            chunk_size = min(self.batch_size, 50)  # Smaller chunks for better error handling

            for i in range(0, len(actions), chunk_size):
                chunk = actions[i:i + chunk_size]

                try:
                    success, failed_items = helpers.bulk(
                        self.es,
                        chunk,
                        chunk_size=chunk_size,
                        raise_on_error=False,
                        raise_on_exception=False
                    )

                    success_count += success

                    # Handle failed items
                    if failed_items:
                        for failed_item in failed_items:
                            if isinstance(failed_item, dict):
                                # Extract error information
                                op_type = list(failed_item.keys())[0]  # 'create', 'index', etc.
                                error_info = failed_item[op_type]
                                doc_id = error_info.get("_id", "unknown")
                                error_reason = error_info.get("error", {}).get("reason", "Unknown error")

                                # Skip version conflicts (document already exists)
                                if "version_conflict" in error_reason or "document_already_exists" in error_reason:
                                    logger.debug(f"Document already exists (skipped): {doc_id}")
                                    skipped_existing += 1
                                else:
                                    failed_count += 1
                                    errors.append({"doc_id": doc_id, "error": error_reason})
                                    self.failed_documents.append(doc_id)
                                    logger.error(f"Failed to index document {doc_id}: {error_reason}")

                except Exception as chunk_error:
                    logger.error(f"Error processing chunk: {chunk_error}")
                    failed_count += len(chunk)
                    for action in chunk:
                        doc_id = action.get("_id", "unknown")
                        errors.append({"doc_id": doc_id, "error": str(chunk_error)})
                        self.failed_documents.append(doc_id)

            self.total_indexed += success_count
            self.total_failed += failed_count

            logger.info(f"Bulk indexed: {success_count} successful, {failed_count} failed, "
                       f"{skipped_existing + skipped_no_embedding} skipped")

            return {
                "indexed": success_count,
                "failed": failed_count,
                "errors": errors,
                "skipped": skipped_existing + skipped_no_embedding
            }

        except Exception as e:
            # Log the error
            logger.error(f"Bulk indexing error: {e}")

            # Mark all documents as failed
            failed_count = len(documents_to_index)
            self.total_failed += failed_count

            errors = []
            for doc in documents_to_index:
                doc_id = doc.get("patent_id", doc.get("id", "unknown"))
                errors.append({"doc_id": doc_id, "error": str(e)})
                self.failed_documents.append(doc_id)

            return {
                "indexed": 0,
                "failed": failed_count,
                "errors": errors,
                "skipped": skipped_existing + skipped_no_embedding
            }

    def search_similar_documents(
        self,
        query_vector: List[float],
        k: int = 10,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        vector_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score
            filters: Additional filters to apply
            vector_field: Target vector field (default: summary vector)

        Returns:
            List of similar documents
        """
        target_field = vector_field or self.summary_vector_field

        # Build the query
        query = {
            "knn": {
                "field": target_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": min(k * 10, self.max_num_candidates)
            },
            "min_score": min_score
        }

        # Add filters if provided
        if filters:
            query["query"] = {"bool": {"filter": filters}}

        # Execute search
        try:
            response = self.es.search(
                index=self.index_name,
                body=query,
                size=k
            )

            # Extract results
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                result["_id"] = hit["_id"]

                # Remove vector fields from result to reduce size
                result.pop(self.summary_vector_field, None)
                result.pop(self.claims1_vector_field, None)

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        k: int = 10,
        text_weight: float = 0.4,
        vector_weight: float = 0.6,
        min_score: float = 0.0,
        vector_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search.

        Args:
            query_text: Text query
            query_vector: Query embedding vector
            k: Number of results to return
            text_weight: Weight for text search
            vector_weight: Weight for vector search

        Returns:
            List of search results
        """
        target_field = vector_field or self.summary_vector_field

        # Build query
        should_clauses = []

        # Add text search weighting multiple fields to better capture domain terms
        if query_text:
            should_clauses.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^3",
                            "summary^2",
                            "claims_text^3",
                            "keywords^2",
                            "classification_fi",
                            "f_term",
                            "topics",
                        ],
                        "type": "best_fields",
                        "boost": text_weight,
                        "operator": "and",
                    }
                }
            )

        # Build the main query
        query_body = {
            "query": {
                "bool": {
                    "should": should_clauses
                }
            },
            "size": k,
            "min_score": min_score,
        }

        # Add vector search if vector is provided
        if query_vector:
            query_body["knn"] = {
                "field": target_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": min(k * 10, self.max_num_candidates),
                "boost": vector_weight,
            }

        # Execute search
        try:
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )

            # Extract results with metadata
            results: List[Dict[str, Any]] = []
            for hit in response["hits"]["hits"]:
                source = dict(hit["_source"])
                source.pop(self.summary_vector_field, None)
                source.pop(self.claims1_vector_field, None)
                results.append(
                    {
                        "patent_id": hit["_id"],
                        "_score": hit["_score"],
                        "_vector_field": target_field,
                        "document": source,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def dual_vector_search(
        self,
        query_vector: List[float],
        k: int = 10,
        summary_weight: Optional[float] = None,
        claims_weight: Optional[float] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run separate KNN searches against summary/claims vectors and merge the results.

        Args:
            query_vector: Embedding vector for the query
            k: Maximum number of combined results to return
            summary_weight: Weight applied to the summary score (defaults to env)
            claims_weight: Weight applied to the claims score (defaults to env)
            min_score: Minimum cosine similarity threshold per vector search

        Returns:
            Dict containing combined hits and hit counts per vector field
        """
        summary_weight = summary_weight if summary_weight is not None else self.summary_vector_weight
        claims_weight = claims_weight if claims_weight is not None else self.claims_vector_weight
        min_score = min_score if min_score is not None else self.min_vector_score

        def _run_knn(field_name: str) -> List[Dict[str, Any]]:
            """Execute a KNN search for the provided field."""
            body = {
                "knn": {
                    "field": field_name,
                    "query_vector": query_vector,
                    "k": k,
                    "num_candidates": min(k * 10, self.max_num_candidates),
                },
                "min_score": min_score,
            }
            response = self.es.search(
                index=self.index_name,
                body=body,
                size=k,
            )
            return response["hits"]["hits"]

        def _clean_source(source: Dict[str, Any]) -> Dict[str, Any]:
            """Remove heavy vector payloads from the source."""
            cleaned = dict(source)
            cleaned.pop(self.summary_vector_field, None)
            cleaned.pop(self.claims1_vector_field, None)
            return cleaned

        try:
            summary_hits = _run_knn(self.summary_vector_field)
            claims_hits = _run_knn(self.claims1_vector_field)
        except Exception as exc:
            logger.error(f"Error executing dual vector search: {exc}")
            return {
                "combined_hits": [],
                "summary_hits_count": 0,
                "claims_hits_count": 0,
                "error": str(exc),
            }

        combined: Dict[str, Dict[str, Any]] = {}

        # Merge summary hits
        for rank, hit in enumerate(summary_hits, start=1):
            doc_id = hit["_id"]
            score = hit["_score"]
            entry = combined.setdefault(
                doc_id,
                {
                    "patent_id": doc_id,
                    "combined_score": 0.0,
                    "summary_score": None,
                    "claims_score": None,
                    "summary_rank": None,
                    "claims_rank": None,
                    "document": _clean_source(hit["_source"]),
                    "sources": [],
                },
            )
            entry["summary_score"] = score
            entry["summary_rank"] = rank
            entry["combined_score"] += summary_weight * score
            entry["sources"].append(
                {
                    "vector_field": self.summary_vector_field,
                    "rank": rank,
                    "score": score,
                    "weight": summary_weight,
                    "weighted_score": summary_weight * score,
                }
            )

        # Merge claims hits
        for rank, hit in enumerate(claims_hits, start=1):
            doc_id = hit["_id"]
            score = hit["_score"]
            entry = combined.setdefault(
                doc_id,
                {
                    "patent_id": doc_id,
                    "combined_score": 0.0,
                    "summary_score": None,
                    "claims_score": None,
                    "summary_rank": None,
                    "claims_rank": None,
                    "document": _clean_source(hit["_source"]),
                    "sources": [],
                },
            )
            entry["claims_score"] = score
            entry["claims_rank"] = rank
            entry["combined_score"] += claims_weight * score
            if "document" not in entry or not entry["document"]:
                entry["document"] = _clean_source(hit["_source"])
            entry["sources"].append(
                {
                    "vector_field": self.claims1_vector_field,
                    "rank": rank,
                    "score": score,
                    "weight": claims_weight,
                    "weighted_score": claims_weight * score,
                }
            )

        combined_hits = sorted(
            combined.values(),
            key=lambda item: (
                -float(item.get("combined_score", 0.0)),
                -float(item.get("summary_score") or 0.0),
                -float(item.get("claims_score") or 0.0),
            ),
        )[:k]

        return {
            "combined_hits": combined_hits,
            "summary_hits_count": len(summary_hits),
            "claims_hits_count": len(claims_hits),
        }

    def get_document_by_id(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its patent ID.

        Args:
            patent_id: Patent ID

        Returns:
            Document if found, None otherwise
        """
        # Check connection first
        if not self.is_connected():
            logger.warning(f"Elasticsearch connection is closed, skipping document retrieval for {patent_id}")
            return None

        try:
            response = self.es.get(
                index=self.index_name,
                id=patent_id
            )

            if response.get("found"):
                return response["_source"]
            return None

        except Exception as e:
            logger.error(f"Error retrieving document {patent_id}: {e}")
            return None

    def update_document(self, patent_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document in the index.

        Args:
            patent_id: Patent ID
            updates: Fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.es.update(
                index=self.index_name,
                id=patent_id,
                body={"doc": updates}
            )

            return response.get("result") == "updated"

        except Exception as e:
            logger.error(f"Error updating document {patent_id}: {e}")
            return False

    def delete_document(self, patent_id: str) -> bool:
        """
        Delete a document from the index.

        Args:
            patent_id: Patent ID

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.es.delete(
                index=self.index_name,
                id=patent_id
            )

            return response.get("result") == "deleted"

        except Exception as e:
            logger.error(f"Error deleting document {patent_id}: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        # Check connection first
        if not self.is_connected():
            logger.warning("Elasticsearch connection is closed, returning cached stats only")
            return {
                "index_name": self.index_name,
                "document_count": 0,
                "size_in_bytes": 0,
                "total_indexed": self.total_indexed,
                "total_failed": self.total_failed,
                "failed_documents": self.failed_documents,
                "connection_closed": True
            }

        try:
            # Get index stats
            stats = self.es.indices.stats(index=self.index_name)

            # Get document count
            count = self.es.count(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "total_indexed": self.total_indexed,
                "total_failed": self.total_failed,
                "failed_documents": self.failed_documents
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "index_name": self.index_name,
                "document_count": 0,
                "size_in_bytes": 0,
                "total_indexed": self.total_indexed,
                "total_failed": self.total_failed,
                "failed_documents": self.failed_documents,
                "error": str(e)
            }

    def get_random_document_ids(self, sample_size: int = 1000) -> List[str]:
        """
        Get random document IDs from the index for sampling.

        Args:
            sample_size: Number of document IDs to retrieve

        Returns:
            List of document IDs that exist in Elasticsearch
        """
        if not self.is_connected():
            logger.warning("Elasticsearch connection is closed, cannot retrieve document IDs")
            return []

        try:
            # Use scroll to get document IDs efficiently
            query = {
                "query": {"match_all": {}},
                "_source": False,  # Only get IDs, not the full documents
                "size": min(sample_size, 10000)  # Limit to prevent memory issues
            }

            response = self.es.search(
                index=self.index_name,
                body=query,
                scroll='2m'
            )

            document_ids = []

            # Get IDs from first batch
            for hit in response['hits']['hits']:
                document_ids.append(hit['_id'])

            # If we need more documents and there are more available, continue scrolling
            scroll_id = response.get('_scroll_id')
            while len(document_ids) < sample_size and scroll_id and len(response['hits']['hits']) > 0:
                response = self.es.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )

                for hit in response['hits']['hits']:
                    document_ids.append(hit['_id'])
                    if len(document_ids) >= sample_size:
                        break

            # Clear scroll
            if scroll_id:
                try:
                    self.es.clear_scroll(scroll_id=scroll_id)
                except Exception as e:
                    logger.debug(f"Error clearing scroll: {e}")

            # Shuffle and return requested sample size
            import random
            random.shuffle(document_ids)
            return document_ids[:sample_size]

        except Exception as e:
            logger.error(f"Error getting random document IDs: {e}")
            return []

    def is_connected(self) -> bool:
        """
        Check if Elasticsearch connection is active.

        Returns:
            True if connected, False otherwise
        """
        try:
            return self.es.ping()
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
            return False

    def close(self):
        """Close the Elasticsearch connection."""
        try:
            if hasattr(self, 'es') and self.es:
                self.es.close()
                logger.info("Elasticsearch connection closed")
        except Exception as e:
            logger.warning(f"Error during Elasticsearch connection close: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize indexer
    indexer = ElasticsearchIndexer()

    # Create index
    indexer.create_index(force_recreate=False)

    # Sample document with vector
    sample_doc = {
        "patent_id": "2010000001",
        "title": "バリカン式刈刃装置",
        "summary": "本発明は、バリカン式の刈刃装置に関する。",
        indexer.vector_field: [0.1] * indexer.vector_dims,  # Dummy vector
        "filing_date": "20080611",
        "publication_date": "20100107",
        "classification_ipc": ["A61B8/00"],
        "embedding_model": "text-embedding-3-large",
        "embedding_generated": True
    }

    # Index document
    success = indexer.index_document(sample_doc)
    print(f"Document indexed: {success}")

    # Get stats
    stats = indexer.get_index_stats()
    print(f"Index stats: {json.dumps(stats, indent=2)}")

    indexer.close()
