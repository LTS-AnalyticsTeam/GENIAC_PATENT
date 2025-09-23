"""
Azure OpenAI Embedding Processor with Batch Support
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import AzureOpenAI

from .utils.rate_limiter import BatchRateLimiter
from .utils.token_counter import TokenCounter

load_dotenv()

logger = logging.getLogger(__name__)


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

            # Extract summaries for embedding
            summaries = [doc.get("summary", "") for doc in batch]

            # Generate embeddings for the batch
            embeddings = await self.generate_batch_embeddings(summaries)

            # Add embeddings to documents
            for doc, embedding in zip(batch, embeddings):
                if embedding:
                    doc["summary_vector"] = embedding
                    doc["embedding_model"] = self.model
                    doc["embedding_generated"] = True
                    self.total_processed += 1
                else:
                    doc["embedding_generated"] = False
                    self.total_failed += 1
                    self.failed_documents.append(doc.get("patent_id", "unknown"))
                    logger.error(f"Failed to generate embedding for document: "
                                f"{doc.get('patent_id', 'unknown')}")

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
            summaries = [doc.get("summary", "") for doc in batch]
            embeddings = await self.generate_batch_embeddings(summaries)

            for doc, embedding in zip(batch, embeddings):
                if embedding:
                    doc["summary_vector"] = embedding
                    doc["embedding_model"] = self.model
                    doc["embedding_generated"] = True
                else:
                    doc["embedding_generated"] = False

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
            print(f"Document {doc['patent_id']}: Embedding generated "
                  f"(dimension: {len(doc['summary_vector'])})")
        else:
            print(f"Document {doc['patent_id']}: Failed to generate embedding")

    # Print statistics
    stats = processor.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
