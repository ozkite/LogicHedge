"""
Advanced Hyperliquid client with features from Allora_HyperLiquid_AutoTradeBot
and Perp-Dex-Trading-Bot.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hmac
import hashlib
import time
from web3 import Web3

from .types import Order, OrderSide, OrderType, Position, Balance
from .utils import HyperliquidAPIError

logger = logging.getLogger(__name__)

class AdvancedHyperliquidClient:
    """
    Advanced Hyperliquid client with:
    - Order book management (from Perp-Dex-Trading-Bot)
    - Position hedging
    - Funding rate arbitrage
    - Batch operations
    """
    
    def __init__(self, private_key: str, testnet: bool = False):
        self.private_key = private_key
        self.testnet = testnet
        self.base_url = "https://api.hyperliquid.xyz"
        if testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
            
        self.w3 = Web3()
        self.account = self.w3.eth.account.from_key(private_key)
        self.address = self.account.address
        
        # State tracking (inspired by Perp-Dex-Trading-Bot)
        self.order_books = {}
        self.positions = {}
        self.open_orders = {}
        self.funding_rates = {}
        
        # Connection pools
        self.session = None
        self.ws_client = None
        
    async def initialize(self):
        """Initialize client connections"""
        import aiohttp
        self.session = aiohttp.ClientSession()
        
        # Initialize WebSocket connection
        await self._init_websocket()
        
        # Start market data sync
        asyncio.create_task(self._sync_market_data())
        
    async def _init_websocket(self):
        """Initialize WebSocket connection for real-time data"""
        import websockets
        
        ws_url = self.base_url.replace("https", "wss") + "/ws"
        self.ws_client = await websockets.connect(ws_url)
        
        # Subscribe to order book updates
        await self.ws_client.send(json.dumps({
            "method": "subscribe",
            "subscription": {
                "type": "orderbook",
                "symbol": "all"
            }
        }))
        
        # Start message handler
        asyncio.create_task(self._handle_websocket_messages())
        
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.ws_client:
                data = json.loads(message)
                await self._process_ws_message(data)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
    async def _process_ws_message(self, data: Dict[str, Any]):
        """Process WebSocket messages"""
        msg_type = data.get("type")
        
        if msg_type == "orderbook":
            symbol = data.get("symbol")
            self.order_books[symbol] = {
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        elif msg_type == "trade":
            symbol = data.get("symbol")
            # Update local order book with trades
            await self._update_orderbook_from_trade(symbol, data)
            
        elif msg_type == "position":
            position_data = data.get("data")
            self.positions[position_data["symbol"]] = position_data
            
    async def _sync_market_data(self):
        """Sync market data periodically"""
        while True:
            try:
                # Sync funding rates
                await self._sync_funding_rates()
                
                # Sync positions
                await self._sync_positions()
                
                # Sync open orders
                await self._sync_open_orders()
                
                await asyncio.sleep(5)  # Sync every 5 seconds
                
            except Exception as e:
                logger.error(f"Market data sync error: {e}")
                await asyncio.sleep(10)
                
    async def _sync_funding_rates(self):
        """Sync funding rates for all symbols"""
        try:
            # This is a simplified version - implement actual API call
            symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
            for symbol in symbols:
                # Get funding rate from API
                rate = await self._get_funding_rate(symbol)
                if rate:
                    self.funding_rates[symbol] = rate
        except Exception as e:
            logger.error(f"Funding rate sync error: {e}")
            
    async def place_batch_orders(self, orders: List[Order]) -> List[Dict[str, Any]]:
        """
        Place multiple orders in batch (inspired by NoFx batch operations)
        """
        batch_data = []
        for order in orders:
            order_msg = self._create_order_message(order)
            signature = self._sign_message(order_msg)
            
            batch_data.append({
                "order": order_msg,
                "signature": signature
            })
            
        # Send batch request
        endpoint = f"{self.base_url}/batch/orders"
        async with self.session.post(endpoint, json=batch_data) as response:
            if response.status != 200:
                raise HyperliquidAPIError(f"Batch order failed: {await response.text()}")
                
            return await response.json()
            
    def _create_order_message(self, order: Order) -> Dict[str, Any]:
        """Create order message for signing"""
        return {
            "action": "order",
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price else None,
            "timeInForce": order.time_in_force.value,
            "reduceOnly": order.reduce_only,
            "timestamp": int(time.time() * 1000)
        }
        
    def _sign_message(self, message: Dict[str, Any]) -> str:
        """Sign message with private key"""
        message_str = json.dumps(message, separators=(',', ':'))
        signed = self.account.sign_message(
            self.w3.eth.account.messages.encode_defunct(text=message_str)
        )
        return signed.signature.hex()
        
    async def get_mid_price(self, symbol: str) -> float:
        """Get mid price from order book"""
        if symbol not in self.order_books:
            return None
            
        book = self.order_books[symbol]
        if not book["bids"] or not book["asks"]:
            return None
            
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        
        return (best_bid + best_ask) / 2
        
    async def get_order_book_imbalance(self, symbol: str, depth: int = 10) -> float:
        """
        Calculate order book imbalance (inspired by NoFx trading signals)
        Returns value between -1 (all sells) and 1 (all buys)
        """
        if symbol not in self.order_books:
            return 0.0
            
        book = self.order_books[symbol]
        bids = book["bids"][:depth]
        asks = book["asks"][:depth]
        
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        
        if bid_volume + ask_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
    async def hedge_position(self, symbol: str, hedge_ratio: float = 1.0):
        """
        Hedge position with opposite side on different venue
        Inspired by Perp-Dex-Trading-Bot hedging strategies
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        size = float(position["size"])
        
        if size == 0:
            return
            
        # Calculate hedge size
        hedge_size = abs(size) * hedge_ratio
        hedge_side = "SELL" if size > 0 else "BUY"
        
        # Create hedge order
        hedge_order = Order(
            symbol=symbol,
            side=OrderSide[hedge_side],
            order_type=OrderType.MARKET,
            quantity=hedge_size,
            reduce_only=True
        )
        
        # Place hedge order on different venue (simplified)
        # In production, this would route to a different exchange
        return await self.place_order(hedge_order)
        
    async def execute_twap(self, symbol: str, side: str, total_quantity: float, 
                          duration_minutes: int = 5, slices: int = 10) -> List[Dict[str, Any]]:
        """
        Execute Time-Weighted Average Price order
        Inspired by NoFx execution algorithms
        """
        slice_quantity = total_quantity / slices
        slice_interval = (duration_minutes * 60) / slices
        
        executions = []
        
        for i in range(slices):
            # Wait for next slice
            await asyncio.sleep(slice_interval)
            
            # Get current market price
            mid_price = await self.get_mid_price(symbol)
            if not mid_price:
                continue
                
            # Place market order for slice
            order = Order(
                symbol=symbol,
                side=OrderSide[side],
                order_type=OrderType.MARKET,
                quantity=slice_quantity
            )
            
            try:
                result = await self.place_order(order)
                executions.append(result)
                logger.info(f"TWAP slice {i+1}/{slices} executed: {slice_quantity} {symbol}")
            except Exception as e:
                logger.error(f"TWAP slice {i+1} failed: {e}")
                
        return executions
        
    async def close_position_with_vwap(self, symbol: str, lookback_minutes: int = 5):
        """
        Close position using Volume-Weighted Average Price
        Inspired by Perp-Dex-Trading-Bot exit strategies
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        size = float(position["size"])
        
        if size == 0:
            return
            
        # Get recent trades for VWAP calculation
        trades = await self.get_recent_trades(symbol, lookback_minutes)
        
        if not trades:
            # Fall back to market order
            return await self.close_position(symbol)
            
        # Calculate VWAP
        total_volume = sum(trade["volume"] for trade in trades)
        total_value = sum(trade["price"] * trade["volume"] for trade in trades)
        
        if total_volume == 0:
            return await self.close_position(symbol)
            
        vwap = total_value / total_volume
        
        # Place limit order at VWAP
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL if size > 0 else OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=abs(size),
            price=vwap,
            reduce_only=True
        )
        
        return await self.place_order(order)
        
    async def get_recent_trades(self, symbol: str, lookback_minutes: int) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol"""
        endpoint = f"{self.base_url}/trades"
        params = {
            "symbol": symbol,
            "limit": 100
        }
        
        async with self.session.get(endpoint, params=params) as response:
            if response.status != 200:
                return []
                
            trades = await response.json()
            
            # Filter by time
            cutoff = time.time() - (lookback_minutes * 60)
            recent_trades = [
                trade for trade in trades 
                if trade.get("timestamp", 0) > cutoff * 1000
            ]
            
            return recent_trades
