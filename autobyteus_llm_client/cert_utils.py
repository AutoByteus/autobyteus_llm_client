import hashlib
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

class CertificateError(Exception):
    """Custom exception for certificate-related errors"""
    pass

def get_certificate_info(cert_path: Union[str, Path]) -> Dict:
    """
    Get detailed information about a certificate.
    
    Args:
        cert_path: Path to the certificate file
        
    Returns:
        Dictionary containing certificate details
        
    Raises:
        CertificateError: If certificate cannot be read or parsed
    """
    try:
        cert_path = Path(cert_path)
        cert_data = cert_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        
        # Get fingerprint matching OpenSSL's output
        cert_der = cert.fingerprint(hashes.SHA256())
        fingerprint = ':'.join([f'{b:02X}' for b in cert_der])
        
        now = datetime.utcnow()
        days_until_expiry = (cert.not_valid_after - now).days
        
        return {
            'subject': cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value,
            'issuer': cert.issuer.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value,
            'valid_from': cert.not_valid_before,
            'valid_until': cert.not_valid_after,
            'fingerprint': fingerprint,
            'is_valid': now < cert.not_valid_after and now > cert.not_valid_before,
            'days_until_expiry': days_until_expiry,
            'cert_data': cert_data,
            'cert': cert
        }
    except Exception as e:
        raise CertificateError(f"Failed to get certificate info: {str(e)}")

def verify_certificate(
    cert_path: Union[str, Path],
    expected_fingerprint: Optional[str] = None,
    warn_expiry_days: int = 30
) -> Dict:
    """
    Verify the certificate's validity and fingerprint.
    
    Args:
        cert_path: Path to the certificate file
        expected_fingerprint: Expected SHA256 fingerprint (optional)
        warn_expiry_days: Days threshold for expiration warning
        
    Returns:
        Dictionary containing certificate details and verification results
        
    Raises:
        CertificateError: If any verification check fails
    """
    try:
        info = get_certificate_info(cert_path)
        
        # Check basic validity
        if not info['is_valid']:
            if datetime.utcnow() < info['valid_from']:
                raise CertificateError("Certificate is not yet valid")
            else:
                raise CertificateError(
                    f"Certificate has expired on {info['valid_until'].strftime('%Y-%m-%d')}"
                )
        
        # Verify fingerprint if provided
        if expected_fingerprint:
            # Remove any spaces and ensure consistent case
            expected = expected_fingerprint.replace(' ', '').upper()
            actual = info['fingerprint'].replace(' ', '')
            if actual != expected:
                raise CertificateError(
                    f"Certificate fingerprint mismatch. "
                    f"Expected: {expected}\n"
                    f"Got: {actual}"
                )
            logger.info("Certificate fingerprint verified successfully")
        else:
            logger.warning(
                "Certificate fingerprint verification skipped. "
                "Set AUTOBYTEUS_CERT_FINGERPRINT environment variable to enable this security feature. "
                f"Current certificate fingerprint: {info['fingerprint']}"
            )
        
        # Log certificate details
        logger.info(f"Certificate valid from {info['valid_from']} to {info['valid_until']}")
        logger.info(f"Certificate fingerprint (SHA256): {info['fingerprint']}")
        logger.info(f"Certificate subject: {info['subject']}")
        
        # Check for upcoming expiration
        if info['days_until_expiry'] <= warn_expiry_days:
            logger.warning(
                f"Certificate will expire in {info['days_until_expiry']} days "
                f"on {info['valid_until'].strftime('%Y-%m-%d')}"
            )
        
        return info
        
    except CertificateError:
        raise
    except Exception as e:
        raise CertificateError(f"Certificate verification failed: {str(e)}")