"""
Azure OpenAI Embedding Processor with Batch Support
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from .utils.rate_limiter import BatchRateLimiter
from .utils.token_counter import TokenCounter
from .utils.text_extractor import extract_claims1_text, extract_summary_text

load_dotenv()

logger = logging.getLogger(__name__)

SUMMARY_VECTOR_FIELD = os.getenv("ES_SUMMARY_VECTOR_FIELD", "summary_vector")
CLAIMS1_VECTOR_FIELD = os.getenv("ES_CLAIMS1_VECTOR_FIELD", "claims1_vector")


class EmbeddingProcessor:
    """Process text embeddings using OpenAI API with batch support."""

    def __init__(self):
        """Initialize the embedding processor."""
        # Azure OpenAI configuration
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://patent-openai.openai.azure.com/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-large")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY must be set in environment variables")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key
        )

        self.model = self.deployment  # Use deployment name as model
        self.summary_vector_field = SUMMARY_VECTOR_FIELD
        self.claims1_vector_field = CLAIMS1_VECTOR_FIELD

        # Rate limiting configuration
        max_tokens_per_min = int(os.getenv("OPENAI_MAX_TOKENS_PER_MIN", "1000000"))
        batch_size = int(os.getenv("OPENAI_BATCH_SIZE", "20"))
        concurrent_workers = int(os.getenv("CONCURRENT_WORKERS", "5"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

        # Initialize utilities
        self.token_counter = TokenCounter(self.model)
        self.rate_limiter = BatchRateLimiter(
            max_concurrent_workers=concurrent_workers,
            max_tokens_per_minute=max_tokens_per_min,
            batch_size=batch_size
        )

        self.max_retries = max_retries
        self.batch_size = batch_size

        # Statistics
        self.total_processed = 0
        self.total_failed = 0
        self.failed_documents = []

    async def generate_embedding(
        self,
        text: str,
        retry_count: int = 0
    ) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for
            retry_count: Current retry attempt

        Returns:
            Embedding vector or None if failed
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            return None

        try:
            # Count tokens
            token_count = self.token_counter.count_tokens(text)

            # Wait for rate limit if needed
            await self.rate_limiter.wait_for_token_budget(token_count)

            # Generate embedding using Azure OpenAI
            response = self.client.embeddings.create(
                model=self.deployment,
                input=text
            )

            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                await asyncio.sleep(wait_time)
                return await self.generate_embedding(text, retry_count + 1)
            return None

    async def generate_batch_embeddings(
        self,
        texts: List[str]
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings (None for failed items)
        """
        if not texts:
            return []

        try:
            # Calculate total tokens for the batch
            total_tokens = self.token_counter.count_batch_tokens(texts)

            # Wait for rate limit if needed
            await self.rate_limiter.wait_for_token_budget(total_tokens)

            # Generate embeddings in batch using Azure OpenAI
            response = self.client.embeddings.create(
                model=self.deployment,
                input=texts
            )

            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                logger.warning(f"Rate limit hit for batch. Processing individually: {e}")
            else:
                logger.error(f"Error generating batch embeddings: {e}")
            # Fall back to individual processing
            embeddings = []
            for text in texts:
                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)
            return embeddings

    async def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of documents to add embeddings.

        Args:
            documents: List of patent documents

        Returns:
            Documents with added embedding fields
        """
        if not documents:
            return []

        # Split documents into optimal batches based on token count
        doc_batches = self.token_counter.calculate_batches_for_documents(
            documents,
            max_tokens_per_batch=8191  # Max for text-embedding-3-large
        )

        logger.info(f"Processing {len(documents)} documents in {len(doc_batches)} batches")

        processed_documents = []

        for batch_idx, batch in enumerate(doc_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(doc_batches)} "
                       f"with {len(batch)} documents")

            # Extract text for embedding
            summary_texts = [extract_summary_text(doc) for doc in batch]
            claims_texts = [extract_claims1_text(doc) for doc in batch]

            # Generate embeddings for summaryと請求項1
            summary_embeddings = await self.generate_batch_embeddings(summary_texts)
            claims_embeddings = await self.generate_batch_embeddings(claims_texts)

            # Add embeddings to documents
            for doc, summary_embedding, claims_embedding in zip(batch, summary_embeddings, claims_embeddings):
                summary_generated = bool(summary_embedding)
                claims_generated = bool(claims_embedding)

                if summary_generated:
                    doc[self.summary_vector_field] = summary_embedding
                else:
                    doc.pop(self.summary_vector_field, None)

                if claims_generated:
                    doc[self.claims1_vector_field] = claims_embedding
                else:
                    doc.pop(self.claims1_vector_field, None)

                doc["embedding_model"] = self.model
                doc["embedding_generated_summary"] = summary_generated
                doc["embedding_generated_claims1"] = claims_generated
                doc["embedding_generated"] = summary_generated and claims_generated

                if doc["embedding_generated"]:
                    self.total_processed += 1
                else:
                    self.total_failed += 1
                    self.failed_documents.append(doc.get("patent_id", "unknown"))
                    if not summary_generated or not claims_generated:
                        logger.warning(
                            "Partial embedding generation failure for document %s "
                            "(summary=%s, claims1=%s)",
                            doc.get("patent_id", "unknown"),
                            "ok" if summary_generated else "missing",
                            "ok" if claims_generated else "missing",
                        )

                processed_documents.append(doc)

        return processed_documents

    async def process_documents_parallel(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process documents in parallel with rate limiting.

        Args:
            documents: List of patent documents

        Returns:
            Documents with added embedding fields
        """
        if not documents:
            return []

        # Split into batches
        doc_batches = self.token_counter.calculate_batches_for_documents(
            documents,
            max_tokens_per_batch=8191
        )

        logger.info(f"Processing {len(documents)} documents in {len(doc_batches)} "
                   f"batches with parallel processing")

        # Process batches in parallel
        async def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Process a single batch of documents."""
            summary_texts = [extract_summary_text(doc) for doc in batch]
            claims_texts = [extract_claims1_text(doc) for doc in batch]

            summary_embeddings = await self.generate_batch_embeddings(summary_texts)
            claims_embeddings = await self.generate_batch_embeddings(claims_texts)

            for doc, summary_embedding, claims_embedding in zip(batch, summary_embeddings, claims_embeddings):
                summary_generated = bool(summary_embedding)
                claims_generated = bool(claims_embedding)

                if summary_generated:
                    doc[self.summary_vector_field] = summary_embedding
                else:
                    doc.pop(self.summary_vector_field, None)

                if claims_generated:
                    doc[self.claims1_vector_field] = claims_embedding
                else:
                    doc.pop(self.claims1_vector_field, None)

                doc["embedding_model"] = self.model
                doc["embedding_generated_summary"] = summary_generated
                doc["embedding_generated_claims1"] = claims_generated
                doc["embedding_generated"] = summary_generated and claims_generated

            return batch

        # Create tasks for parallel processing
        tasks = []
        for batch in doc_batches:
            # Calculate token count for the batch
            token_count = sum(
                self.token_counter.estimate_document_tokens(doc)
                for doc in batch
            )

            # Create task with rate limiting
            task = self.rate_limiter.process_with_rate_limit(
                process_batch,
                batch,
                token_count=token_count
            )
            tasks.append(task)

        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        processed_documents = []
        for batch_result in batch_results:
            processed_documents.extend(batch_result)

        # Update statistics
        for doc in processed_documents:
            if doc.get("embedding_generated"):
                self.total_processed += 1
            else:
                self.total_failed += 1
                self.failed_documents.append(doc.get("patent_id", "unknown"))
                logger.warning(
                    "Partial embedding generation failure for document %s "
                    "(summary=%s, claims1=%s)",
                    doc.get("patent_id", "unknown"),
                    "ok" if doc.get("embedding_generated_summary") else "missing",
                    "ok" if doc.get("embedding_generated_claims1") else "missing",
                )

        return processed_documents

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "failed_documents": self.failed_documents,
            "model": self.model,
            "rate_limiter_status": self.rate_limiter.get_status()
        }

    def reset_statistics(self):
        """Reset processing statistics."""
        self.total_processed = 0
        self.total_failed = 0
        self.failed_documents = []


# Example usage
async def main():
    """Example usage of the embedding processor."""
    logging.basicConfig(level=logging.INFO)

    # Initialize processor
    processor = EmbeddingProcessor()

    # Sample documents
    documents = [
        {
            "patent_id": "2010000001",
            "title": "バリカン式刈刃装置",
            "summary": "本発明は、バリカン式の刈刃装置に関する。"
        },
        {
            "patent_id": "2010000002",
            "title": "画像処理装置",
            "summary": "本発明は、画像処理装置および画像処理方法に関する。"
        }
    ]

    # Process documents
    processed = await processor.process_documents_parallel(documents)

    # Check results
    for doc in processed:
        if doc.get("embedding_generated"):
            summary_vec = doc.get(processor.summary_vector_field, [])
            claims_vec = doc.get(processor.claims1_vector_field, [])
            print(
                f"Document {doc['patent_id']}: summary_vec={len(summary_vec)} dims, "
                f"claims1_vec={len(claims_vec)} dims"
            )
        else:
            print(f"Document {doc['patent_id']}: Failed to generate embedding")

    # Print statistics
    stats = processor.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
