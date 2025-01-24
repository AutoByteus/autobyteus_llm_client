import os
import pytest
import pytest_asyncio
from autobyteus_llm_client.client import AutobyteusClient
from typing import AsyncGenerator
from pathlib import Path
from dotenv import load_dotenv

def load_test_env():
    """Load test environment variables from .env.test file"""
    project_root = Path(__file__).parent.parent
    env_test_path = project_root / '.env.test'
    
    if not env_test_path.exists():
        raise FileNotFoundError(f"Test environment file not found: {env_test_path}")
    
    load_dotenv(env_test_path)

@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Session-wide fixture to set and restore test environment variables"""
    original_env = os.environ.copy()
    
    # Load test environment variables
    load_test_env()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AutobyteusClient, None]:
    """Async fixture providing configured test client"""
    client = AutobyteusClient()
    try:
        yield client
    finally:
        await client.close()