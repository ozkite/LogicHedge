"""
WebSocket client for Hyperliquid real-time data.
"""

import asyncio
import json
import logging
from typing import Callable, Dict, Any, Optional
import websockets

logger = logging.getLogger(__name__)

class HyperliquidWebSocket:
    """WebSocket client for Hyperliquid real-time market data"""
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        if testnet:
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
            
        self.websocket = None
        self.connected = False
        self.subscriptions = set()
        self.callbacks = {
            "ticker": [],
            "orderbook": [],
            "trades": [],
            "positions": [],
            "orders": []
        }
        
    async def connect(self):
        """Connect to Hyperliquid WebSocket"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            logger.info("Connected to Hyperliquid WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to Hyperliquid WebSocket: {e}")
            raise
            
    async def subscribe(self, channel: str, symbol: str):
        """Subscribe to market data channel"""
        subscription = {
            "method": "subscribe",
            "subscription": {
                "type": channel,
                "symbol": symbol
            }
        }
        
        await self.websocket.send(json.dumps(subscription))
        self.subscriptions.add((channel, symbol))
        logger.debug(f"Subscribed to {channel}:{symbol}")
        
    async def unsubscribe(self, channel: str, symbol: str):
        """Unsubscribe from market data channel"""
        subscription = {
            "method": "unsubscribe",
            "subscription": {
                "type": channel,
                "symbol": symbol
            }
        }
        
        await self.websocket.send(json.dumps(subscription))
        self.subscriptions.remove((channel, symbol))
        
    def register_callback(self, channel: str, callback: Callable):
        """Register callback for specific channel"""
        if channel in self.callbacks:
            self.callbacks[channel].append(callback)
            
    async def listen(self):
        """Listen for incoming messages"""
        async for message in self.websocket:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message: {message}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        
        if message_type == "ticker":
            await self._notify_callbacks("ticker", data)
        elif message_type == "orderbook":
            await self._notify_callbacks("orderbook", data)
        elif message_type == "trade":
            await self._notify_callbacks("trades", data)
        elif message_type == "position":
            await self._notify_callbacks("positions", data)
        elif message_type == "order":
            await self._notify_callbacks("orders", data)
            
    async def _notify_callbacks(self, channel: str, data: Dict[str, Any]):
        """Notify all registered callbacks"""
        for callback in self.callbacks.get(channel, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
                
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Hyperliquid WebSocket disconnected")
