# Autobyteus LLM Client

Python async client library for interacting with the Autobyteus LLM API.

## Installation

```bash
pip install autobyteus-llm-client
```

For development/testing:
```bash
pip install -e .[test]
```

## Configuration

Set required environment variables:
```bash
export AUTOBYTEUS_API_KEY="your_api_key_here"
export AUTOBYTEUS_SERVER_URL="http://localhost:8000"  # Optional, defaults to localhost
```

## Usage

```python
from autobyteus_llm_client import AutobyteusClient

async def main():
    # Initialize client
    client = AutobyteusClient()
    
    # Get available models
    models = await client.get_available_models()
    print(f"Available models: {models}")
    
    # Send a message
    response = await client.send_message(
        conversation_id="test_conv",
        model_name="gpt-4",
        user_message="Hello, world!"
    )
    print(f"Response: {response}")
    
    # Clean up conversation
    await client.cleanup("test_conv")
    await client.close()

# Run the async main function
import asyncio
asyncio.run(main())
```

## Streaming Responses

```python
async def stream_example():
    client = AutobyteusClient()
    try:
        async for chunk in client.stream_message(
            conversation_id="stream_conv",
            model_name="gpt-4",
            user_message="Stream this response"
        ):
            print(f"Received chunk: {chunk}")
    finally:
        await client.cleanup("stream_conv")
        await client.close()
```

## Testing

1. Create a `.env.test` file with test credentials:
```bash
AUTOBYTEUS_API_KEY="test_key"
AUTOBYTEUS_SERVER_URL="http://test-server:8000"
```

2. Run tests:
```bash
pytest -v autobyteus_llm_client/tests/
```

## Error Handling

The client will raise `RuntimeError` for API errors. Always wrap calls in try/except blocks:

```python
try:
    response = await client.send_message(...)
except RuntimeError as e:
    print(f"API error occurred: {str(e)}")
```

## License

MIT License
