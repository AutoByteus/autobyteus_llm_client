import pytest
import httpx
import json
import os
from autobyteus_llm_client.client import AutobyteusClient
from typing import Dict, Any

@pytest.mark.asyncio
async def test_get_available_models(client: AutobyteusClient):
    """Test model listing functionality"""
    response = await client.get_available_models()
    
    assert isinstance(response, dict), "Response should be a dictionary"
    assert 'models' in response, "Response should contain 'models' key"
    assert isinstance(response['models'], list), "Models should be a list"
    
    if response['models']:
        model = response['models'][0]
        assert all(key in model for key in ('name', 'value', 'config')), \
            "Model entry missing required fields"
        assert isinstance(model['config'], dict), "Config should be a dictionary"
        assert any(key in model['config'] for key in ('token_limit', 'rate_limit')), \
            "Config missing required configuration keys"
        if 'pricing_config' in model['config']:
            pricing = model['config']['pricing_config']
            assert 'input_token_pricing' in pricing, "Missing input token pricing"
            assert 'output_token_pricing' in pricing, "Missing output token pricing"

@pytest.mark.asyncio
async def test_send_message(client: AutobyteusClient):
    """Test basic message sending with file attachments"""
    models = await client.get_available_models()
    if not models['models']:
        pytest.skip("No available models to test")
    
    model = models['models'][0]
    response = await client.send_message(
        conversation_id="test-conv-1",
        model_name=model['value'],
        user_message="hello, how are you doing",
        file_paths=[],
        user_message_index=0
    )
    
    assert 'response' in response, "Missing response field"
    assert isinstance(response['response'], str), "Response should be a string"
    
    if 'token_usage' in response:
        usage = response['token_usage']
        assert all(key in usage for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')), \
            "Token usage missing required fields"
        assert all(isinstance(usage[key], int) for key in usage), "Token counts should be integers"
        assert usage['total_tokens'] > 0, "Token usage should be positive"

@pytest.mark.asyncio
async def test_stream_message(client: AutobyteusClient):
    """Test streaming functionality with full validation"""
    models = await client.get_available_models()
    if not models['models']:
        pytest.skip("No available models to test")
    
    model = models['models'][0]
    stream = client.stream_message(
        conversation_id="test-conv-3",
        model_name=model['value'],
        user_message="Hello, how are you doing",
        file_paths=[],
        user_message_index=0
    )
    
    received_chunks = []
    async for chunk in stream:
        assert isinstance(chunk, dict), "Chunk should be a dictionary"
        assert 'content' in chunk, "Chunk missing content field"
        
        # Validate intermediate chunks have no token usage
        if not chunk.get('is_complete', False):
            assert chunk.get('token_usage') is None, \
                "Intermediate chunks should not contain token usage data"
        
        received_chunks.append(chunk)

    assert len(received_chunks) > 0, "Should receive at least one chunk"
    
    final_chunk = received_chunks[-1]
    assert final_chunk.get('is_complete', False), "Last chunk should mark completion"
    
    # Validate final chunk contains token usage
    assert 'token_usage' in final_chunk, "Final chunk missing token usage data"
    usage = final_chunk['token_usage']
    assert all(key in usage for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')), \
        "Token usage missing required fields in final chunk"
    assert all(isinstance(usage[key], int) for key in usage), \
        "Token counts should be integers"
    assert usage['total_tokens'] > 0, "Total tokens should be positive"
    assert usage['prompt_tokens'] <= usage['total_tokens'], \
        "Prompt tokens should not exceed total"
    assert usage['completion_tokens'] <= usage['total_tokens'], \
        "Completion tokens should not exceed total"

@pytest.mark.asyncio
async def test_cleanup(client: AutobyteusClient):
    """Test conversation cleanup workflow with existence check"""
    models = await client.get_available_models()
    if not models['models']:
        pytest.skip("No available models to test")
    
    model = models['models'][0]
    test_conv_id = "test-conv-3"
    
    await client.send_message(
        conversation_id=test_conv_id,
        model_name=model['value'],
        user_message="Cleanup test message"
    )
    
    response = await client.cleanup(test_conv_id)
    assert isinstance(response, dict), "Response should be a dictionary"
    assert response.get('success', False), "Cleanup should report success"
    assert 'message' in response, "Cleanup response should include message"
    
    with pytest.raises(RuntimeError) as exc_info:
        await client.send_message(
            conversation_id=test_conv_id,
            model_name=model['value'],
            user_message="Should fail after cleanup"
        )
    
    assert "404" in str(exc_info.value), "Should error when using cleaned up conversation ID"

@pytest.mark.asyncio
async def test_error_handling(client: AutobyteusClient):
    """Test client error handling with specific error messages"""
    with pytest.raises(RuntimeError) as exc_info:
        await client.send_message(
            conversation_id="test-conv-4",
            model_name="non-existent-model",
            user_message="This should fail"
        )
    
    error_detail = str(exc_info.value).lower()
    assert "not found" in error_detail, "Error message should indicate model not found"
    assert "404" in error_detail, "Error should contain status code information"

@pytest.mark.asyncio
async def test_missing_required_fields(client: AutobyteusClient):
    """Test validation of required fields with detailed assertions"""
    models = await client.get_available_models()
    if not models['models']:
        pytest.skip("No available models to test")
    
    model = models['models'][0]
    with pytest.raises(RuntimeError) as exc_info:
        await client.send_message(
            conversation_id="test-conv-5",
            model_name=model['value'],
            user_message=""  # Invalid empty message
        )
    
    error_data = json.loads(str(exc_info.value))
    assert any("user_message" in err['loc'] for err in error_data.get('detail', [])), \
        "Error should reference user_message field"

@pytest.mark.asyncio
async def test_invalid_api_key():
    """Test authentication failure scenario with fresh client"""
    original_key = os.environ['AUTOBYTEUS_API_KEY']
    os.environ['AUTOBYTEUS_API_KEY'] = 'invalid-key'
    
    try:
        # Create new client with updated environment
        async with AutobyteusClient() as client:
            with pytest.raises(RuntimeError) as exc_info:
                await client.get_available_models()
            
            error_detail = str(exc_info.value).lower()
            assert "401" in error_detail, "Should return 401 for invalid API key"
            assert "unauthorized" in error_detail, "Error should indicate authorization failure"
    finally:
        os.environ['AUTOBYTEUS_API_KEY'] = original_key

@pytest.mark.asyncio
async def test_stream_error_propagation(client: AutobyteusClient):
    """Test error propagation in streaming responses"""
    models = await client.get_available_models()
    if not models['models']:
        pytest.skip("No available models to test")
    
    model = models['models'][0]
    stream = client.stream_message(
        conversation_id="test-conv-6",
        model_name=model['value'],
        user_message="Valid message",
        user_message_index=3
    )
    
    # Simulate mid-stream error
    async with httpx.AsyncClient() as hacker_client:
        await hacker_client.post(
            "http://localhost:8000/cleanup",
            json={"conversation_id": "test-conv-6"},
            headers={"AUTOBYTEUS_API_KEY": "test-key-123"}
        )
    
    with pytest.raises(RuntimeError) as exc_info:
        async for chunk in stream:
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
    
    assert "cleanup" in str(exc_info.value).lower(), "Should propagate stream interruption error"
