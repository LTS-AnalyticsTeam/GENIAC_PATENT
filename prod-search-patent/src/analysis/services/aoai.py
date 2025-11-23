from __future__ import annotations
from typing import Any, Dict, List, Optional
import httpx
import asyncio
import time


class RateLimiter:
    """Simple token bucket rate limiter"""
    def __init__(self, calls_per_minute: int = 20):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last_call)
            self.last_call_time = time.time()


class AOAIClient:
    """
    Azure OpenAI Chat Completions の薄ラッパ。
    429エラー対策のリトライロジックとレート制限を実装。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        calls_per_minute: int = 2,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        if not (self.endpoint and self.api_key and self.api_version):
            raise RuntimeError("Azure OpenAI settings incomplete")
        
        # レート制限用
        self.rate_limiter = RateLimiter(calls_per_minute)
        
    async def _call_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, str],
        payload: Dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> httpx.Response:
        """リトライロジック with exponential backoff"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # レート制限を適用
                await self.rate_limiter.acquire()
                
                # API呼び出し
                response = await client.post(
                    url, 
                    headers=headers, 
                    params=params, 
                    json=payload,
                    timeout=httpx.Timeout(60.0, connect=10.0)
                )
                
                # 成功した場合はそのまま返す
                if response.status_code < 400:
                    return response
                
                # 429エラーの場合
                if response.status_code == 429:
                    # Retry-Afterヘッダーがあればその値を使用
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        # Exponential backoff: 1秒, 2秒, 4秒...
                        wait_time = base_delay * (2 ** attempt)
                    
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                    last_exception = httpx.HTTPStatusError(
                        f"Rate limit exceeded (429)", 
                        request=response.request, 
                        response=response
                    )
                    continue
                
                # その他のHTTPエラー
                response.raise_for_status()
                
            except httpx.TimeoutException as e:
                print(f"Timeout error on attempt {attempt + 1}/{max_retries}: {e}")
                wait_time = base_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
                
            except httpx.HTTPStatusError as e:
                # 429以外のHTTPエラー
                if e.response.status_code >= 500:
                    # サーバーエラーの場合はリトライ
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Server error {e.response.status_code}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    # 4xx系エラー（429以外）はリトライしない
                    raise
                    
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                continue
        
        # 全てのリトライが失敗した場合
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Failed after {max_retries} attempts")

    async def chat(
        self,
        deployment: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        if not deployment:
            raise ValueError("deployment must be provided")

        url = f"{self.endpoint}/openai/deployments/{deployment}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient() as client:
            response = await self._call_with_retry(
                client, url, headers, params, payload, max_retries
            )
            return response.json()
