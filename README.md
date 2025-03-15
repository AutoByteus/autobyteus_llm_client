# Autobyteus LLM Client

Async Python client for Autobyteus LLM API with HTTPS support.

## Installation

```bash
pip install autobyteus_llm_client
```

## Certificate Setup

1. Create certificates directory:
```bash
mkdir -p certificates
```

2. Copy the server's certificate:
```bash
cp path/to/server/certificates/cert.pem certificates/
```

3. Get certificate fingerprint (optional but recommended):
```bash
openssl x509 -in certificates/cert.pem -fingerprint -sha256 -noout
```

## Configuration

Set environment variables:
```bash
# Required
export AUTOBYTEUS_API_KEY='your-api-key'

# Optional (defaults shown)
export AUTOBYTEUS_LLM_SERVER_URL='https://api.autobyteus.com:8443'
export AUTOBYTEUS_CERT_FINGERPRINT='your-certificate-fingerprint'  # Optional but recommended
```

## Usage

```python
from autobyteus_llm_client import AutobyteusClient

async def main():
    # Initialize client (automatically uses certificate from certificates/cert.pem)
    client = AutobyteusClient()
    
    try:
        # Get available models
        models = await client.get_available_models()
        print(f"Available models: {models}")
        
        # Send a message
        response = await client.send_message(
            conversation_id="conv123",
            model_name="gpt-4",
            user_message="Hello!"
        )
        print(f"Response: {response}")
        
    finally:
        await client.close()
```

## Security Features

1. Certificate Verification
   - Automatic certificate validation
   - Certificate expiration checking
   - Optional fingerprint verification
   - Path validation and security checks

2. SSL/TLS Security
   - HTTPS communication
   - Certificate-based authentication
   - Secure default configuration

## Development

### Requirements
- Python 3.8 or higher
- httpx
- cryptography

### Installing Development Dependencies
```bash
pip install -e ".[test]"
```

### Running Tests
```bash
pytest
```

## Building and Publishing

### Build Package
```bash
python -m build
```

### Publish to Test PyPI
```bash
python -m twine upload --repository testpypi dist/*
```

### Publish to Production PyPI
```bash
python -m twine upload dist/*
```

## Troubleshooting

1. Certificate Issues
   - Verify certificate location (should be in `certificates/cert.pem`)
   - Check certificate expiration
   - Verify fingerprint if enabled

2. Connection Issues
   - Verify server URL and port
   - Check certificate validity
   - Ensure API key is set correctly

## License

This project is licensed under the MIT License.
