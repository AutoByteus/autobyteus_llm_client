import os
import httpx
import logging
import json
from typing import Dict, Optional, AsyncGenerator, Any
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class AutobyteusClient:
    DEFAULT_SERVER_URL = "http://localhost:8000"
    API_KEY_HEADER = "AUTOBYTEUS_API_KEY"
    API_KEY_ENV_VAR = "AUTOBYTEUS_API_KEY"
    
    def __init__(self):
        self.server_url = os.getenv('AUTOBYTEUS_SERVER_URL', self.DEFAULT_SERVER_URL)
        self.api_key = os.getenv(self.API_KEY_ENV_VAR)
        
        if not self.api_key:
            raise ValueError(
                f"{self.API_KEY_ENV_VAR} environment variable is required. "
                "Please set it before initializing the client."
            )
        
        # Async client for normal operations
        self.async_client = httpx.AsyncClient(
            verify=True,
            headers={self.API_KEY_HEADER: self.api_key},
            timeout=httpx.Timeout(10.0)
        )
        
        # Sync client for discovery operations
        self.sync_client = httpx.Client(
            verify=True,
            headers={self.API_KEY_HEADER: self.api_key},
            timeout=10.0
        )
        
        logger.info(f"Initialized Autobyteus client with server URL: {self.server_url}")

    def get_available_models_sync(self) -> Dict[str, Any]:
        """Synchronous model discovery for factory initialization"""
        try:
            response = self.sync_client.get(urljoin(self.server_url, "/models"))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            error_detail = self._extract_error_detail(e)
            logger.error(f"Sync model fetch error: {error_detail}")
            raise RuntimeError(error_detail) from e

    async def get_available_models(self) -> Dict[str, Any]:
        """Async model discovery for other use cases"""
        try:
            response = await self.async_client.get(urljoin(self.server_url, "/models"))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            error_detail = self._extract_error_detail(e)
            logger.error(f"Async model fetch error: {error_detail}")
            raise RuntimeError(error_detail) from e

    async def send_message(
        self,
        conversation_id: str,
        model_name: str,
        user_message: str,
        file_paths: Optional[list] = None,
        user_message_index: int = 0
    ) -> Dict[str, Any]:
        """Send a message and get a response"""
        try:
            data = {
                "conversation_id": conversation_id,
                "model_name": model_name,
                "user_message": user_message,
                "file_paths": file_paths or [],
                "user_message_index": user_message_index
            }
            response = await self.async_client.post(
                urljoin(self.server_url, "/send-message"),
                json=data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            error_detail = self._extract_error_detail(e)
            logger.error(f"Error sending message: {error_detail}")
            raise RuntimeError(error_detail) from e

    async def stream_message(
        self,
        conversation_id: str,
        model_name: str,
        user_message: str,
        file_paths: Optional[list] = None,
        user_message_index: int = 0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a message and get responses"""
        try:
            data = {
                "conversation_id": conversation_id,
                "model_name": model_name,
                "user_message": user_message,
                "file_paths": file_paths or [],
                "user_message_index": user_message_index
            }
            
            async with self.async_client.stream(
                "POST",
                urljoin(self.server_url, "/stream-message"),
                json=data
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            if 'error' in chunk:
                                raise RuntimeError(chunk['error'])
                            yield chunk
                        except json.JSONDecodeError:
                            logger.error("Failed to parse stream chunk")
                            raise RuntimeError("Invalid stream response format")

        except httpx.HTTPError as e:
            error_detail = self._extract_error_detail(e)
            logger.error(f"Stream error: {error_detail}")
            raise RuntimeError(error_detail) from e
        
    async def cleanup(self, conversation_id: str) -> Dict[str, Any]:
        """Clean up a conversation"""
        try:
            response = await self.async_client.post(
                urljoin(self.server_url, "/cleanup"),
                json={"conversation_id": conversation_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            error_detail = self._extract_error_detail(e)
            logger.error(f"Cleanup error: {error_detail}")
            raise RuntimeError(error_detail) from e

    async def close(self):
        """Close both clients"""
        await self.async_client.aclose()
        self.sync_client.close()

    def _extract_error_detail(self, error: httpx.HTTPError) -> str:
        """Extract error details from HTTP response"""
        if isinstance(error, httpx.HTTPStatusError):
            try:
                response_data = error.response.json()
                return response_data.get('detail', str(error))
            except json.JSONDecodeError:
                return error.response.text or str(error)
        return str(error)