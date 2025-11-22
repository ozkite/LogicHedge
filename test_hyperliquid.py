#!/usr/bin/env python3
"""
Simple test script for Hyperliquid connector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from logichedge.exchanges.dex.hyperliquid.client import HyperliquidClient

def test_connection():
    """Test basic connection to Hyperliquid"""
    print("Testing Hyperliquid connection...")
    
    # Initialize client (public endpoints don't need API key)
    client = HyperliquidClient(testnet=True)  # Use testnet for testing
    
    try:
        # Test public endpoints
        info = client.get_exchange_info()
        print("✅ Exchange info retrieved successfully")
        print(f"Exchange: {info.get('exchange', 'Unknown')}")
        
        # Test orderbook for a popular pair
        orderbook = client.get_orderbook("BTC-USDT")
        print("✅ Orderbook data retrieved successfully")
        
        print("\n🎉 Hyperliquid connector is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_connection()
