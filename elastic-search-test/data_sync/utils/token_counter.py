"""
Token counting utility for OpenAI API rate limiting
"""
import logging
from typing import Any, Dict, List

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Utility class for counting tokens in text for OpenAI models."""

    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initialize token counter with specific model encoding.

        Args:
            model_name: Name of the OpenAI model
        """
        self.model_name = model_name

        # Get the encoding for the model
        try:
            # For embedding models, use cl100k_base encoding
            if "embedding" in model_name:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found. Using cl100k_base encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def count_batch_tokens(self, texts: List[str]) -> int:
        """
        Count total tokens in a batch of texts.

        Args:
            texts: List of texts to count tokens for

        Returns:
            Total number of tokens
        """
        total_tokens = 0
        for text in texts:
            total_tokens += self.count_tokens(text)
        return total_tokens

    def estimate_document_tokens(self, document: Dict[str, Any]) -> int:
        """
        Estimate tokens for a patent document.

        Args:
            document: Patent document dictionary

        Returns:
            Estimated number of tokens
        """
        # Count tokens in the summary (main field for embedding)
        summary_tokens = self.count_tokens(document.get("summary", ""))

        # If we need to embed other fields in the future
        # title_tokens = self.count_tokens(document.get("title", ""))
        # description_tokens = self.count_tokens(document.get("description", ""))

        return summary_tokens

    def split_texts_by_token_limit(
        self,
        texts: List[str],
        max_tokens_per_batch: int = 8191
    ) -> List[List[str]]:
        """
        Split texts into batches based on token limit.

        Args:
            texts: List of texts to split
            max_tokens_per_batch: Maximum tokens per batch

        Returns:
            List of text batches
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.count_tokens(text)

            # If single text exceeds limit, truncate it
            if text_tokens > max_tokens_per_batch:
                logger.warning(f"Text exceeds token limit ({text_tokens} > {max_tokens_per_batch}). Truncating.")
                # Truncate the text to fit within limit
                truncated_text = self._truncate_text(text, max_tokens_per_batch)
                batches.append([truncated_text])
                continue

            # Check if adding this text would exceed the limit
            if current_tokens + text_tokens > max_tokens_per_batch:
                # Save current batch and start new one
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                # Add to current batch
                current_batch.append(text)
                current_tokens += text_tokens

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)

        return truncated_text

    def calculate_batches_for_documents(
        self,
        documents: List[Dict[str, Any]],
        max_tokens_per_batch: int = 8191
    ) -> List[List[Dict[str, Any]]]:
        """
        Calculate optimal batches for documents based on token limits.

        Args:
            documents: List of patent documents
            max_tokens_per_batch: Maximum tokens per batch

        Returns:
            List of document batches
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = self.estimate_document_tokens(doc)

            # If single document exceeds limit, put it in its own batch
            if doc_tokens > max_tokens_per_batch:
                logger.warning(
                    f"Document {doc.get('patent_id')} exceeds token limit "
                    f"({doc_tokens} > {max_tokens_per_batch})"
                )
                batches.append([doc])
                continue

            # Check if adding this document would exceed the limit
            if current_tokens + doc_tokens > max_tokens_per_batch:
                # Save current batch and start new one
                if current_batch:
                    batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                # Add to current batch
                current_batch.append(doc)
                current_tokens += doc_tokens

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Split {len(documents)} documents into {len(batches)} batches")
        return batches


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize counter
    counter = TokenCounter("text-embedding-3-large")

    # Test token counting
    sample_text = "これは日本語のテキストです。トークン数をカウントします。"
    tokens = counter.count_tokens(sample_text)
    print(f"Sample text tokens: {tokens}")

    # Test batch splitting
    texts = [
        "短いテキスト",
        "これは少し長めのテキストです。" * 10,
        "別のテキスト",
    ]

    batches = counter.split_texts_by_token_limit(texts, max_tokens_per_batch=50)
    print(f"Split into {len(batches)} batches")
    for i, batch in enumerate(batches):
        batch_tokens = counter.count_batch_tokens(batch)
        print(f"Batch {i+1}: {len(batch)} texts, {batch_tokens} tokens")
