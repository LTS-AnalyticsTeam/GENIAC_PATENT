"""
Vector Search API endpoints for patent documents
"""
import logging
import os
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from fastapi import HTTPException
from openai import AzureOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "claims_vector")
CLAIMS_TOP3_FIELD = os.getenv("ES_CLAIMS_TOP3_FIELD", "claims_top3_text")


class VectorSearchRequest(BaseModel):
    """Request model for vector search."""
    query: str
    k: int = 10
    min_score: float = 0.7
    filters: Optional[Dict[str, Any]] = None


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str
    k: int = 10
    rank_constant: int = 60  # RRF rank constant (lower values favor top results)
    filters: Optional[Dict[str, Any]] = None


class SimilarDocumentRequest(BaseModel):
    """Request model for finding similar documents."""
    patent_id: str
    k: int = 10
    min_score: float = 0.7


class VectorSearchService:
    """Service for handling vector search operations."""

    def __init__(self, es_client: Elasticsearch):
        """
        Initialize the vector search service.

        Args:
            es_client: Elasticsearch client instance
        """
        self.es = es_client
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "patent_vectors")

        # Initialize Azure OpenAI client for generating query embeddings
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://patent-openai.openai.azure.com/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-large")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if self.api_key:
            # Clear proxy environment variables to avoid conflicts
            import os as os_module
            for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                if proxy_var in os_module.environ:
                    del os_module.environ[proxy_var]

            # Simple initialization without extra parameters
            self.openai_client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key
            )
            logger.info("Azure OpenAI client initialized successfully")
        else:
            logger.warning("Azure OpenAI API key not found. Vector search will be limited.")
            self.openai_client = None

    async def generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Generate embedding for a query text.

        Args:
            query_text: Text to generate embedding for

        Returns:
            Embedding vector or None if failed
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return None

        try:
            response = self.openai_client.embeddings.create(
                model=self.deployment,
                input=query_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    async def vector_search(
        self,
        query_text: str,
        k: int = 10,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure vector search.

        Args:
            query_text: Query text
            k: Number of results
            min_score: Minimum similarity score
            filters: Additional filters

        Returns:
            List of search results
        """
        # Generate query embedding
        query_vector = await self.generate_query_embedding(query_text)
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")

        # Build the query
        query = {
            "knn": {
                "field": VECTOR_FIELD,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": min(k * 10, 10_000)
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

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "patent_id": hit["_id"],
                    "score": hit["_score"],
                    "title": hit["_source"].get("title", ""),
                    "summary": hit["_source"].get("summary", ""),
                    "filing_date": hit["_source"].get("filing_date", ""),
                    "publication_date": hit["_source"].get("publication_date", ""),
                    "classification_ipc": hit["_source"].get("classification_ipc", []),
                    "keywords": hit["_source"].get("keywords", []),
                    "applicants": hit["_source"].get("applicants", [])
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def hybrid_search(
        self,
        query_text: str,
        k: int = 10,
        rank_constant: int = 60,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Elasticsearch's native RRF (Reciprocal Rank Fusion).

        Args:
            query_text: Query text
            k: Number of results
            rank_constant: RRF rank constant (default 60, lower values favor top results)
            filters: Additional filters

        Returns:
            List of search results
        """
        # Generate query embedding
        query_vector = await self.generate_query_embedding(query_text)
        if not query_vector:
            # Fall back to text-only search if embedding fails
            logger.warning("Failed to generate embedding, falling back to text search")
            return await self._text_only_search(query_text, k, filters)

        # Build RRF hybrid search query
        retrievers = []

        # Add vector search retriever
        knn_retriever = {
            "knn": {
                "field": VECTOR_FIELD,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": min(k * 10, 10_000)
            }
        }

        # Add filters to KNN if provided
        if filters:
            knn_retriever["knn"]["filter"] = filters

        retrievers.append(knn_retriever)

        # Add text search retriever
        text_query = {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^2",
                    "summary",
                    "claims_text^2",
                    CLAIMS_TOP3_FIELD,
                    "keywords",
                    "topics"
                ],
                "type": "best_fields"
            }
        }

        # Build text retriever with filters if provided
        if filters:
            text_retriever = {
                "standard": {
                    "query": {
                        "bool": {
                            "must": text_query,
                            "filter": filters
                        }
                    }
                }
            }
        else:
            text_retriever = {
                "standard": {
                    "query": text_query
                }
            }

        retrievers.append(text_retriever)

        # Build the RRF query
        query_body = {
            "retriever": {
                "rrf": {
                    "retrievers": retrievers,
                    "window_size": k * 2,  # Consider more results for better fusion
                    "rank_constant": rank_constant
                }
            },
            "size": k
        }

        # Execute search
        try:
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "patent_id": hit["_id"],
                    "score": hit["_score"],
                    "title": hit["_source"].get("title", ""),
                    "summary": hit["_source"].get("summary", ""),
                    "filing_date": hit["_source"].get("filing_date", ""),
                    "publication_date": hit["_source"].get("publication_date", ""),
                    "classification_ipc": hit["_source"].get("classification_ipc", []),
                    "keywords": hit["_source"].get("keywords", []),
                    "applicants": hit["_source"].get("applicants", []),
                    "search_type": "hybrid_rrf"
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in hybrid RRF search: {e}")
            # Fall back to legacy hybrid search if RRF is not supported
            if "retriever" in str(e) or "rrf" in str(e):
                logger.warning("RRF not supported, falling back to legacy hybrid search")
                return await self._legacy_hybrid_search(query_text, k, filters)
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def _text_only_search(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform text-only search as fallback.
        """
        query = {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^2",
                    "summary",
                    "claims_text^2",
                    CLAIMS_TOP3_FIELD,
                    "keywords",
                    "topics"
                ],
                "type": "best_fields"
            }
        }

        query_body = {
            "query": query if not filters else {
                "bool": {
                    "must": query,
                    "filter": filters
                }
            },
            "size": k
        }

        try:
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "patent_id": hit["_id"],
                    "score": hit["_score"],
                    "title": hit["_source"].get("title", ""),
                    "summary": hit["_source"].get("summary", ""),
                    "filing_date": hit["_source"].get("filing_date", ""),
                    "publication_date": hit["_source"].get("publication_date", ""),
                    "classification_ipc": hit["_source"].get("classification_ipc", []),
                    "keywords": hit["_source"].get("keywords", []),
                    "applicants": hit["_source"].get("applicants", []),
                    "search_type": "text_only"
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in text-only search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def _legacy_hybrid_search(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Legacy hybrid search using boost parameters (fallback for older Elasticsearch versions).
        """
        query_vector = await self.generate_query_embedding(query_text)

        # Build query with traditional boosting
        should_clauses = [{
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^2",
                    "summary",
                    "claims_text^2",
                    CLAIMS_TOP3_FIELD,
                    "keywords",
                    "topics"
                ],
                "type": "best_fields",
                "boost": 0.3
            }
        }]

        query_body = {
            "query": {
                "bool": {
                    "should": should_clauses
                }
            },
            "size": k
        }

        if filters:
            query_body["query"]["bool"]["filter"] = filters

        if query_vector:
            query_body["knn"] = {
                "field": VECTOR_FIELD,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": min(k * 10, 10_000),
                "boost": 0.7
            }

        try:
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "patent_id": hit["_id"],
                    "score": hit["_score"],
                    "title": hit["_source"].get("title", ""),
                    "summary": hit["_source"].get("summary", ""),
                    "filing_date": hit["_source"].get("filing_date", ""),
                    "publication_date": hit["_source"].get("publication_date", ""),
                    "classification_ipc": hit["_source"].get("classification_ipc", []),
                    "keywords": hit["_source"].get("keywords", []),
                    "applicants": hit["_source"].get("applicants", []),
                    "search_type": "hybrid_legacy"
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in legacy hybrid search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def find_similar_documents(
        self,
        patent_id: str,
        k: int = 10,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given patent.

        Args:
            patent_id: Patent ID to find similar documents for
            k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of similar documents
        """
        try:
            # Get the document and its vector
            doc_response = self.es.get(
                index=self.index_name,
                id=patent_id
            )

            if not doc_response.get("found"):
                raise HTTPException(status_code=404, detail=f"Patent {patent_id} not found")

            source = doc_response["_source"]

            # Check if document has a vector
            if VECTOR_FIELD not in source:
                raise HTTPException(
                    status_code=400,
                    detail=f"Patent {patent_id} does not have an embedding vector"
                )

            query_vector = source[VECTOR_FIELD]

            # Search for similar documents (excluding the source document)
            query = {
                "knn": {
                    "field": VECTOR_FIELD,
                    "query_vector": query_vector,
                    "k": k + 1,  # Get one extra to exclude the source
                    "num_candidates": min((k + 1) * 10, 10_000)
                },
                "min_score": min_score,
                "query": {
                    "bool": {
                        "must_not": {
                            "term": {"_id": patent_id}
                        }
                    }
                }
            }

            response = self.es.search(
                index=self.index_name,
                body=query,
                size=k
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                if hit["_id"] != patent_id:  # Double-check to exclude source
                    result = {
                        "patent_id": hit["_id"],
                        "similarity_score": hit["_score"],
                        "title": hit["_source"].get("title", ""),
                        "summary": hit["_source"].get("summary", ""),
                        "filing_date": hit["_source"].get("filing_date", ""),
                        "publication_date": hit["_source"].get("publication_date", ""),
                        "classification_ipc": hit["_source"].get("classification_ipc", []),
                        "keywords": hit["_source"].get("keywords", []),
                        "applicants": hit["_source"].get("applicants", [])
                    }
                    results.append(result)

            return results[:k]  # Ensure we return exactly k results

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    async def search_by_classification(
        self,
        classification_code: str,
        classification_type: str = "ipc",
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search documents by classification code.

        Args:
            classification_code: Classification code to search
            classification_type: Type of classification (ipc, fi, f_term, theme_code)
            k: Number of results

        Returns:
            List of documents with the classification
        """
        # Map classification type to field name
        field_map = {
            "ipc": "classification_ipc",
            "fi": "classification_fi",
            "f_term": "f_term",
            "theme_code": "theme_code"
        }

        field_name = field_map.get(classification_type)
        if not field_name:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classification type: {classification_type}"
            )

        # Build query
        query = {
            "query": {
                "term": {
                    field_name: classification_code
                }
            },
            "size": k,
            "sort": [
                {"publication_date": {"order": "desc"}}
            ]
        }

        try:
            response = self.es.search(
                index=self.index_name,
                body=query
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "patent_id": hit["_id"],
                    "title": hit["_source"].get("title", ""),
                    "summary": hit["_source"].get("summary", ""),
                    "filing_date": hit["_source"].get("filing_date", ""),
                    "publication_date": hit["_source"].get("publication_date", ""),
                    "classification_ipc": hit["_source"].get("classification_ipc", []),
                    "keywords": hit["_source"].get("keywords", []),
                    "applicants": hit["_source"].get("applicants", [])
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching by classification: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
