"""
Rate limiting utility for API calls
"""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for controlling API request frequency."""

    def __init__(
        self,
        max_requests_per_minute: int = 3000,
        max_tokens_per_minute: int = 1000000
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum number of requests per minute
            max_tokens_per_minute: Maximum number of tokens per minute
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # Track request times and token usage
        self.request_times = []
        self.token_usage = []

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def wait_if_needed(self, tokens: int = 0):
        """
        Wait if rate limit would be exceeded.

        Args:
            tokens: Number of tokens for this request
        """
        async with self.lock:
            current_time = time.time()

            # Clean up old entries (older than 1 minute)
            self._cleanup_old_entries(current_time)

            # Check request rate limit
            if len(self.request_times) >= self.max_requests_per_minute:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = 60 - (current_time - oldest_request)

                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    self._cleanup_old_entries(time.time())

            # Check token rate limit
            total_tokens = sum(self.token_usage)
            if total_tokens + tokens > self.max_tokens_per_minute:
                # Calculate wait time based on token usage
                if self.token_usage:
                    oldest_token_time = self.request_times[0]
                    wait_time = 60 - (current_time - oldest_token_time)

                    if wait_time > 0:
                        logger.info(
                            f"Token limit reached ({total_tokens + tokens} > "
                            f"{self.max_tokens_per_minute}). Waiting {wait_time:.2f} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        # Clean up again after waiting
                        self._cleanup_old_entries(time.time())

            # Record this request
            self.request_times.append(time.time())
            if tokens > 0:
                self.token_usage.append(tokens)

    def _cleanup_old_entries(self, current_time: float):
        """
        Remove entries older than 1 minute.

        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - 60

        # Find the index where entries are still valid
        valid_index = 0
        for i, request_time in enumerate(self.request_times):
            if request_time > cutoff_time:
                valid_index = i
                break
        else:
            # All entries are old
            valid_index = len(self.request_times)

        # Keep only valid entries
        self.request_times = self.request_times[valid_index:]
        self.token_usage = self.token_usage[valid_index:]

    def get_current_usage(self) -> dict:
        """
        Get current usage statistics.

        Returns:
            Dictionary with usage stats
        """
        current_time = time.time()
        self._cleanup_old_entries(current_time)

        return {
            "requests_in_last_minute": len(self.request_times),
            "tokens_in_last_minute": sum(self.token_usage),
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute
        }


class BatchRateLimiter:
    """Rate limiter specifically for batch processing."""

    def __init__(
        self,
        max_concurrent_workers: int = 5,
        max_tokens_per_minute: int = 1000000,
        batch_size: int = 20
    ):
        """
        Initialize batch rate limiter.

        Args:
            max_concurrent_workers: Maximum number of concurrent workers
            max_tokens_per_minute: Maximum tokens per minute
            batch_size: Default batch size
        """
        self.max_concurrent_workers = max_concurrent_workers
        self.max_tokens_per_minute = max_tokens_per_minute
        self.batch_size = batch_size

        # Semaphore for controlling concurrent workers
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)

        # Track token usage
        self.token_times = []
        self.token_counts = []

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a slot for processing."""
        await self.semaphore.acquire()

    def release(self):
        """Release a processing slot."""
        self.semaphore.release()

    async def wait_for_token_budget(self, token_count: int):
        """
        Wait until there's enough token budget.

        Args:
            token_count: Number of tokens needed
        """
        async with self.lock:
            while True:
                current_time = time.time()

                # Clean up old entries
                cutoff_time = current_time - 60
                valid_indices = [
                    i for i, t in enumerate(self.token_times)
                    if t > cutoff_time
                ]

                if valid_indices:
                    self.token_times = [self.token_times[i] for i in valid_indices]
                    self.token_counts = [self.token_counts[i] for i in valid_indices]
                else:
                    self.token_times = []
                    self.token_counts = []

                # Check if we can proceed
                total_tokens = sum(self.token_counts)
                if total_tokens + token_count <= self.max_tokens_per_minute:
                    # Record token usage
                    self.token_times.append(current_time)
                    self.token_counts.append(token_count)
                    break

                # Calculate wait time
                if self.token_times:
                    oldest_time = self.token_times[0]
                    wait_time = 60 - (current_time - oldest_time) + 1

                    logger.info(
                        f"Token budget exceeded ({total_tokens + token_count} > "
                        f"{self.max_tokens_per_minute}). Waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # No previous usage, proceed
                    self.token_times.append(current_time)
                    self.token_counts.append(token_count)
                    break

    async def process_with_rate_limit(self, func, *args, **kwargs):
        """
        Process a function with rate limiting.

        Args:
            func: Async function to call
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function
        """
        await self.acquire()
        try:
            # Get token count from kwargs if provided
            token_count = kwargs.pop('token_count', 0)
            if token_count > 0:
                await self.wait_for_token_budget(token_count)

            # Call the function
            result = await func(*args, **kwargs)
            return result
        finally:
            self.release()

    def get_status(self) -> dict:
        """
        Get current rate limiter status.

        Returns:
            Status dictionary
        """
        current_time = time.time()
        cutoff_time = current_time - 60

        # Calculate current usage
        valid_tokens = [
            count for time_stamp, count in zip(self.token_times, self.token_counts)
            if time_stamp > cutoff_time
        ]

        return {
            "active_workers": self.max_concurrent_workers - self.semaphore._value,
            "max_workers": self.max_concurrent_workers,
            "tokens_used_last_minute": sum(valid_tokens),
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "batch_size": self.batch_size
        }


# Example usage
async def main():
    """Example usage of rate limiters."""

    # Basic rate limiter
    rate_limiter = RateLimiter(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000
    )

    # Simulate API calls
    for i in range(10):
        await rate_limiter.wait_if_needed(tokens=500)
        print(f"Request {i+1} sent")
        await asyncio.sleep(0.1)

    print("\nUsage stats:", rate_limiter.get_current_usage())

    # Batch rate limiter
    batch_limiter = BatchRateLimiter(
        max_concurrent_workers=3,
        max_tokens_per_minute=10000
    )

    async def process_batch(batch_id: int, tokens: int):
        """Simulate batch processing."""
        print(f"Processing batch {batch_id} with {tokens} tokens")
        await asyncio.sleep(1)  # Simulate processing time
        return f"Batch {batch_id} completed"

    # Process multiple batches concurrently
    tasks = []
    for i in range(5):
        task = batch_limiter.process_with_rate_limit(
            process_batch,
            batch_id=i+1,
            tokens=2000,
            token_count=2000
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

    print("\nBatch limiter status:", batch_limiter.get_status())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
