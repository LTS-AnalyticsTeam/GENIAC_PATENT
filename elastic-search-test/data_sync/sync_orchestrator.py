"""
Orchestrator for syncing data from Cosmos DB to Elasticsearch with embeddings
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .cosmos_client import CosmosDBClient
from .elasticsearch_indexer import ElasticsearchIndexer
from .embedding_processor import EmbeddingProcessor

load_dotenv()

logger = logging.getLogger(__name__)


class SyncOrchestrator:
    """Orchestrates the entire sync process from Cosmos DB to Elasticsearch."""

    def __init__(self):
        """Initialize all components."""
        # Initialize clients
        self.cosmos_client = CosmosDBClient()
        self.embedding_processor = EmbeddingProcessor()
        self.es_indexer = ElasticsearchIndexer()

        # Configuration
        self.checkpoint_enabled = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
        self.checkpoint_file = os.getenv("CHECKPOINT_FILE", "./data/checkpoint.json")
        self.enable_incremental = os.getenv("ENABLE_INCREMENTAL_SYNC", "true").lower() == "true"

        # Statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_documents": 0,
            "documents_processed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
            "embeddings_generated": 0,
            "embeddings_failed": 0,
            "errors": []
        }

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.

        Returns:
            Checkpoint data if exists, None otherwise
        """
        if not self.checkpoint_enabled:
            return None

        checkpoint_path = Path(self.checkpoint_file)
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        Save checkpoint to file with validation.

        Args:
            checkpoint_data: Data to save
        """
        if not self.checkpoint_enabled:
            return

        checkpoint_path = Path(self.checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Add validation and additional metadata
        enhanced_checkpoint = {
            **checkpoint_data,
            "checkpoint_version": "2.0",
            "saved_at": datetime.utcnow().isoformat(),
            "process_id": os.getpid(),
            "validation": {
                "documents_processed_valid": checkpoint_data.get("stats", {}).get("documents_processed", 0) >= 0,
                "no_negative_counts": all(
                    v >= 0 for v in [
                        checkpoint_data.get("stats", {}).get("documents_indexed", 0),
                        checkpoint_data.get("stats", {}).get("documents_failed", 0),
                        checkpoint_data.get("stats", {}).get("embeddings_generated", 0),
                        checkpoint_data.get("stats", {}).get("embeddings_failed", 0)
                    ]
                )
            }
        }

        try:
            # Write to temporary file first, then rename for atomic operation
            temp_file = checkpoint_path.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(enhanced_checkpoint, f, indent=2, default=str)

            # Atomic rename
            temp_file.rename(checkpoint_path)
            logger.info(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Clean up temp file if it exists
            temp_file = checkpoint_path.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as cleanup_error:
                    logger.warning(f"Could not clean up temp file: {cleanup_error}")

    async def sync_all_documents(self, force_recreate_index: bool = False, resume_from_checkpoint: bool = True):
        """
        Sync all documents from Cosmos DB to Elasticsearch with improved error handling.

        Args:
            force_recreate_index: If True, recreate the Elasticsearch index
            resume_from_checkpoint: If True, attempt to resume from last checkpoint
        """
        logger.info("Starting full document sync")
        self.stats["start_time"] = datetime.utcnow()

        # Check for existing checkpoint
        checkpoint = None
        if resume_from_checkpoint and not force_recreate_index:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                logger.info(f"Found checkpoint from {checkpoint.get('timestamp')}")
                # Restore statistics from checkpoint
                if "stats" in checkpoint:
                    self.stats.update(checkpoint["stats"])
                    self.stats["start_time"] = datetime.utcnow()  # Reset start time for this session

        try:
            # Create or verify Elasticsearch index
            self.es_indexer.create_index(force_recreate=force_recreate_index)

            # Get total document count
            total_count = self.cosmos_client.get_total_document_count()
            self.stats["total_documents"] = total_count
            logger.info(f"Total documents to process: {total_count}")

            # Determine starting point
            start_batch = 0
            if checkpoint and not force_recreate_index:
                start_batch = checkpoint.get("batch_num", 0)
                logger.info(f"Resuming from batch {start_batch}")

            # Process documents in batches
            batch_num = 0
            processed_batches = 0

            for cosmos_batch in self.cosmos_client.get_all_documents(batch_size=100):
                batch_num += 1

                # Skip batches if resuming from checkpoint
                if batch_num <= start_batch:
                    continue

                processed_batches += 1
                logger.info(f"Processing batch {batch_num} with {len(cosmos_batch)} documents")

                try:
                    # Process embeddings with error handling
                    processed_docs = await self._process_batch_with_retry(cosmos_batch)

                    # Update statistics
                    batch_processed = 0
                    batch_embeddings_generated = 0
                    batch_embeddings_failed = 0

                    for doc in processed_docs:
                        batch_processed += 1
                        if doc.get("embedding_generated"):
                            batch_embeddings_generated += 1
                        else:
                            batch_embeddings_failed += 1

                    # Index to Elasticsearch with duplicate prevention
                    index_result = self.es_indexer.bulk_index_documents(processed_docs, skip_existing=True)

                    # Update statistics
                    self.stats["documents_processed"] += batch_processed
                    self.stats["embeddings_generated"] += batch_embeddings_generated
                    self.stats["embeddings_failed"] += batch_embeddings_failed
                    self.stats["documents_indexed"] += index_result["indexed"]
                    self.stats["documents_failed"] += index_result["failed"]

                    # Log batch results
                    logger.info(f"Batch {batch_num} results: "
                               f"processed={batch_processed}, "
                               f"indexed={index_result['indexed']}, "
                               f"failed={index_result['failed']}, "
                               f"skipped={index_result.get('skipped', 0)}")

                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_num}: {batch_error}")
                    self.stats["errors"].append(f"Batch {batch_num}: {str(batch_error)}")
                    # Continue with next batch instead of failing completely
                    continue

                # Save checkpoint after each successful batch
                if self.checkpoint_enabled:
                    checkpoint_data = {
                        "batch_num": batch_num,
                        "last_processed_id": processed_docs[-1].get("patent_id") if processed_docs else None,
                        "timestamp": datetime.utcnow().isoformat(),
                        "stats": self.stats.copy()
                    }
                    self.save_checkpoint(checkpoint_data)

                # Log progress
                progress = (self.stats["documents_processed"] / total_count) * 100 if total_count > 0 else 0
                logger.info(f"Progress: {progress:.1f}% ({self.stats['documents_processed']}/{total_count})")

            self.stats["end_time"] = datetime.utcnow()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info(f"Full sync completed in {duration:.1f} seconds")
            self._log_statistics()

            # データ整合性チェックを実行（クリーンアップ前に）
            logger.info("🔍 データ整合性チェックを実行中...")
            validation_results = await self.validate_data_integrity()

            if 'error' not in validation_results:
                logger.info(f"   Cosmos DB総数: {validation_results['cosmos_total']:,}件")
                logger.info(f"   Elasticsearch総数: {validation_results['elasticsearch_total']:,}件")
                logger.info(f"   数値一致: {'✅' if validation_results['count_match'] else '❌'}")

                if not validation_results['count_match']:
                    sample_val = validation_results['sample_validation']
                    logger.warning(f"   サンプル検証: {sample_val['checked']}件中{sample_val['valid']}件が一致")
                    if validation_results['missing_documents']:
                        logger.warning(f"   不足ドキュメント例: {validation_results['missing_documents'][:5]}")
            else:
                logger.error(f"   ❌ 整合性チェックエラー: {validation_results['error']}")

            # Save final checkpoint
            if self.checkpoint_enabled:
                final_checkpoint = {
                    "batch_num": batch_num,
                    "completed": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "stats": self.stats.copy(),
                    "validation_results": validation_results
                }
                self.save_checkpoint(final_checkpoint)

        except Exception as e:
            logger.error(f"Error during full sync: {e}")
            self.stats["errors"].append(str(e))
            raise
        finally:
            self._cleanup()

    async def _process_batch_with_retry(self, cosmos_batch, max_retries: int = 3):
        """
        Process a batch of documents with retry logic.

        Args:
            cosmos_batch: Batch of documents from Cosmos DB
            max_retries: Maximum number of retry attempts

        Returns:
            List of processed documents
        """
        for attempt in range(max_retries + 1):
            try:
                processed_docs = await self.embedding_processor.process_documents_parallel(cosmos_batch)
                return processed_docs
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Batch processing failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Batch processing failed after {max_retries + 1} attempts: {e}")
                    # Return documents without embeddings rather than failing completely
                    return [{**doc, "embedding_generated": False} for doc in cosmos_batch]

    async def sync_incremental(self, since_timestamp: Optional[datetime] = None):
        """
        Perform incremental sync of new/updated documents.

        Args:
            since_timestamp: Sync documents modified after this timestamp
        """
        logger.info("Starting incremental sync")
        self.stats["start_time"] = datetime.utcnow()

        try:
            # Load last sync timestamp from checkpoint if not provided
            if since_timestamp is None:
                checkpoint = self.load_checkpoint()
                if checkpoint and "last_sync_timestamp" in checkpoint:
                    since_timestamp = datetime.fromisoformat(checkpoint["last_sync_timestamp"])
                else:
                    # Default to 24 hours ago
                    from datetime import timedelta
                    since_timestamp = datetime.utcnow() - timedelta(hours=24)

            logger.info(f"Syncing documents modified after {since_timestamp}")

            # Process documents
            batch_num = 0
            for cosmos_batch in self.cosmos_client.get_documents_for_incremental_sync(
                since_timestamp,
                batch_size=100
            ):
                batch_num += 1
                logger.info(f"Processing incremental batch {batch_num} with {len(cosmos_batch)} documents")

                # Process embeddings
                processed_docs = await self.embedding_processor.process_documents_parallel(
                    cosmos_batch
                )

                # Update statistics
                self.stats["documents_processed"] += len(processed_docs)
                for doc in processed_docs:
                    if doc.get("embedding_generated"):
                        self.stats["embeddings_generated"] += 1
                    else:
                        self.stats["embeddings_failed"] += 1

                # Index to Elasticsearch
                index_result = self.es_indexer.bulk_index_documents(processed_docs)
                self.stats["documents_indexed"] += index_result["indexed"]
                self.stats["documents_failed"] += index_result["failed"]

            # Save checkpoint with new timestamp
            if self.checkpoint_enabled:
                checkpoint = {
                    "last_sync_timestamp": datetime.utcnow().isoformat(),
                    "documents_synced": self.stats["documents_processed"],
                    "stats": self.stats
                }
                self.save_checkpoint(checkpoint)

            self.stats["end_time"] = datetime.utcnow()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info(f"Incremental sync completed in {duration:.1f} seconds")
            self._log_statistics()

        except Exception as e:
            logger.error(f"Error during incremental sync: {e}")
            self.stats["errors"].append(str(e))
            raise
        finally:
            self._cleanup()

    async def sync_by_date_range(
        self,
        start_date: str,
        end_date: str,
        force_recreate_index: bool = False
    ):
        """
        Sync documents within a specific date range.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            force_recreate_index: If True, recreate the index
        """
        logger.info(f"Starting sync for date range: {start_date} to {end_date}")
        self.stats["start_time"] = datetime.utcnow()

        try:
            # Create or verify Elasticsearch index
            if force_recreate_index:
                self.es_indexer.create_index(force_recreate=True)

            # Process documents
            batch_num = 0
            for cosmos_batch in self.cosmos_client.get_documents_by_date_range(
                start_date,
                end_date,
                batch_size=100
            ):
                batch_num += 1
                logger.info(f"Processing date range batch {batch_num} with {len(cosmos_batch)} documents")

                # Process embeddings
                processed_docs = await self.embedding_processor.process_documents_parallel(
                    cosmos_batch
                )

                # Update statistics
                self.stats["documents_processed"] += len(processed_docs)
                for doc in processed_docs:
                    if doc.get("embedding_generated"):
                        self.stats["embeddings_generated"] += 1
                    else:
                        self.stats["embeddings_failed"] += 1

                # Index to Elasticsearch
                index_result = self.es_indexer.bulk_index_documents(processed_docs)
                self.stats["documents_indexed"] += index_result["indexed"]
                self.stats["documents_failed"] += index_result["failed"]

            self.stats["end_time"] = datetime.utcnow()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info(f"Date range sync completed in {duration:.1f} seconds")
            self._log_statistics()

        except Exception as e:
            logger.error(f"Error during date range sync: {e}")
            self.stats["errors"].append(str(e))
            raise
        finally:
            self._cleanup()

    async def sync_single_document(self, patent_id: str) -> bool:
        """
        Sync a single document by patent ID.

        Args:
            patent_id: Patent ID to sync

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Syncing single document: {patent_id}")

        try:
            # Get document from Cosmos DB
            document = self.cosmos_client.get_document_by_id(patent_id)
            if not document:
                logger.error(f"Document {patent_id} not found in Cosmos DB")
                return False

            # Process embedding
            processed_docs = await self.embedding_processor.process_documents([document])
            if not processed_docs:
                logger.error(f"Failed to process document {patent_id}")
                return False

            processed_doc = processed_docs[0]

            # Index to Elasticsearch
            success = self.es_indexer.index_document(processed_doc)

            if success:
                logger.info(f"Successfully synced document {patent_id}")
            else:
                logger.error(f"Failed to index document {patent_id}")

            return success

        except Exception as e:
            logger.error(f"Error syncing document {patent_id}: {e}")
            return False

    def _log_statistics(self):
        """Log sync statistics."""
        logger.info("=" * 50)
        logger.info("Sync Statistics:")
        logger.info(f"  Total documents: {self.stats['total_documents']}")
        logger.info(f"  Documents processed: {self.stats['documents_processed']}")
        logger.info(f"  Documents indexed: {self.stats['documents_indexed']}")
        logger.info(f"  Documents failed: {self.stats['documents_failed']}")
        logger.info(f"  Embeddings generated: {self.stats['embeddings_generated']}")
        logger.info(f"  Embeddings failed: {self.stats['embeddings_failed']}")

        if self.stats["errors"]:
            logger.info(f"  Errors: {len(self.stats['errors'])}")
            for error in self.stats["errors"][:5]:  # Show first 5 errors
                logger.error(f"    - {error}")

        # Get additional stats from components
        embedding_stats = self.embedding_processor.get_statistics()
        es_stats = self.es_indexer.get_index_stats()

        logger.info(f"  Embedding model: {embedding_stats.get('model')}")
        logger.info(f"  Index document count: {es_stats.get('document_count')}")
        logger.info(f"  Index size: {es_stats.get('size_in_bytes', 0) / (1024*1024):.2f} MB")
        logger.info("=" * 50)

    def _cleanup(self):
        """Clean up resources."""
        try:
            self.cosmos_client.close()
            self.es_indexer.close()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity between Cosmos DB and Elasticsearch.

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting data integrity validation")

        try:
            # Get counts from both sources
            cosmos_count = self.cosmos_client.get_total_document_count()
            es_stats = self.es_indexer.get_index_stats()
            es_count = es_stats.get("document_count", 0)

            # Sample validation - check a few random documents
            validation_results = {
                "cosmos_total": cosmos_count,
                "elasticsearch_total": es_count,
                "count_match": cosmos_count == es_count,
                "missing_documents": [],
                "sample_validation": {"checked": 0, "valid": 0, "invalid": 0}
            }

            # If counts don't match, try to identify missing documents
            if cosmos_count != es_count:
                logger.warning(f"Document count mismatch: Cosmos={cosmos_count}, ES={es_count}")

                # Sample check - validate first 100 documents
                sample_count = 0
                valid_count = 0
                invalid_count = 0

                for cosmos_batch in self.cosmos_client.get_all_documents(batch_size=100):
                    for doc in cosmos_batch:
                        patent_id = doc.get("patent_id")
                        if patent_id:
                            es_doc = self.es_indexer.get_document_by_id(patent_id)
                            sample_count += 1

                            if es_doc:
                                valid_count += 1
                            else:
                                invalid_count += 1
                                validation_results["missing_documents"].append(patent_id)

                        if sample_count >= 100:
                            break

                    if sample_count >= 100:
                        break

                validation_results["sample_validation"] = {
                    "checked": sample_count,
                    "valid": valid_count,
                    "invalid": invalid_count
                }

            logger.info(f"Validation completed: {validation_results}")
            return validation_results

        except Exception as e:
            logger.error(f"Error during data integrity validation: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current sync statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "orchestrator_stats": self.stats,
            "embedding_stats": self.embedding_processor.get_statistics(),
            "elasticsearch_stats": self.es_indexer.get_index_stats()
        }


async def main():
    """Example usage of the sync orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    orchestrator = SyncOrchestrator()

    # Example 1: Full sync
    # await orchestrator.sync_all_documents(force_recreate_index=True)

    # Example 2: Incremental sync
    # await orchestrator.sync_incremental()

    # Example 3: Date range sync
    # await orchestrator.sync_by_date_range("20230101", "20231231")

    # Example 4: Single document sync
    # success = await orchestrator.sync_single_document("2010000001")

    # For testing, let's do a small date range
    await orchestrator.sync_by_date_range("20100101", "20100131")

    # Get final statistics
    stats = orchestrator.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
