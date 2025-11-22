import hashlib
import hmac
import time
import json
from typing import Dict, Any
import requests

def generate_signature(secret: str, message: str) -> str:
    """Generate HMAC signature for Hyperliquid API"""
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def timestamp() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

class HyperliquidAPIError(Exception):
    """Custom exception for Hyperliquid API errors"""
    pass

def handle_response(response: requests.Response) -> Dict[str, Any]:
    """Handle API response and raise appropriate errors"""
    if response.status_code != 200:
        raise HyperliquidAPIError(f"HTTP {response.status_code}: {response.text}")
    
    data = response.json()
    if not data.get('success', True):
        raise HyperliquidAPIError(f"API Error: {data.get('message', 'Unknown error')}")
    
    return data
