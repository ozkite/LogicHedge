"""
Unified market making across multiple exchanges via CedeHub
Place orders on all exchanges simultaneously with shared inventory
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np

class CedeHubMarketMaker:
    """
    Market make on Binance, Bybit, MEXC simultaneously
    Unified inventory management via CedeHub
    """
    
    def __init__(self, cedehub_client, symbols: List[str]):
        self.cedehub = cedehub_client
        self.symbols = symbols
        
        # Unified inventory tracking
        self.unified_inventory = {s: 0.0 for s in symbols}
        self.exchange_inventories = {}
        
        # Market making parameters
        self.spread_bps = 10  # 0.1% spread
        self.order_size_usd = 1000
        self.max_inventory_usd = 10000
        
    async def run_unified_market_making(self):
        """Run market making across all connected exchanges"""
        while True:
            try:
                for symbol in self.symbols:
                    # Get unified fair price from all exchanges
                    fair_price = await self._get_unified_fair_price(symbol)
                    
                    if not fair_price:
                        continue
                        
                    # Calculate bid/ask based on inventory
                    bid_price, ask_price = self._calculate_prices(
                        symbol, fair_price
                    )
                    
                    # Get available inventory across all exchanges
                    available_qty = await self._get_available_inventory(symbol)
                    
                    # Calculate order sizes per exchange
                    exchange_orders = await self._allocate_orders(
                        symbol, bid_price, ask_price, available_qty
                    )
                    
                    # Place orders on all exchanges simultaneously
                    await self._place_unified_orders(exchange_orders)
                    
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Unified MM error: {e}")
                await asyncio.sleep(5)
                
    async def _get_unified_fair_price(self, symbol: str) -> Optional[float]:
        """Get volume-weighted fair price across all exchanges"""
        prices = await self.cedehub._cedehub_request(
            "GET", f"/v1/prices/{symbol.replace('/', '')}"
        )
        
        if not prices.get('exchanges'):
            return None
            
        total_volume = 0
        price_volume = 0
        
        for ex in prices['exchanges']:
            mid = (float(ex['bid']) + float(ex['ask'])) / 2
            volume = float(ex['volume_24h'])
            
            price_volume += mid * volume
            total_volume += volume
            
        return price_volume / total_volume if total_volume > 0 else None
        
    def _calculate_prices(self, symbol: str, fair_price: float) -> Tuple[float, float]:
        """Calculate bid/ask with inventory skew"""
        inventory = self.unified_inventory.get(symbol, 0)
        
        # Calculate skew (negative = short, positive = long)
        skew = inventory / self.max_inventory_usd * fair_price
        
        # Adjust spread based on inventory
        spread_adjustment = skew * 0.0005  # 0.05% per skew unit
        
        bid_price = fair_price * (1 - self.spread_bps/10000/2 - spread_adjustment)
        ask_price = fair_price * (1 + self.spread_bps/10000/2 - spread_adjustment)
        
        return bid_price, ask_price
        
    async def _get_available_inventory(self, symbol: str) -> float:
        """Get available inventory across all exchanges"""
        # Query CedeHub for unified balance
        balance = await self.cedehub._cedehub_request(
            "GET", f"/v1/balance/{symbol.split('/')[0]}"
        )
        
        return float(balance.get('available', 0))
        
    async def _allocate_orders(self, symbol: str, bid: float, ask: float, 
                              available_qty: float) -> Dict[str, List]:
        """Allocate orders to different exchanges based on liquidity"""
        # Get liquidity metrics from CedeHub
        liquidity = await self.cedehub._cedehub_request(
            "GET", f"/v1/liquidity/{symbol.replace('/', '')}"
        )
        
        exchange_orders = {}
        
        for ex_data in liquidity['exchanges']:
            exchange = ex_data['exchange']
            liquidity_score = ex_data['liquidity_score']
            
            # Allocate based on liquidity score
            allocation = liquidity_score / sum(
                e['liquidity_score'] for e in liquidity['exchanges']
            )
            
            bid_size = available_qty * allocation * 0.5  # Half for bids
            ask_size = available_qty * allocation * 0.5  # Half for asks
            
            exchange_orders[exchange] = [
                {"side": "buy", "price": bid, "size": bid_size},
                {"side": "sell", "price": ask, "size": ask_size}
            ]
            
        return exchange_orders
        
    async def _place_unified_orders(self, exchange_orders: Dict[str, List]):
        """Place orders on multiple exchanges via CedeHub"""
        batch_request = {
            "orders": [],
            "execution": "simultaneous",
            "cancel_existing": True
        }
        
        for exchange, orders in exchange_orders.items():
            for order in orders:
                batch_request["orders"].append({
                    "exchange": exchange,
                    "symbol": self.symbols[0],  # Simplified
                    "side": order["side"],
                    "type": "limit",
                    "price": order["price"],
                    "size": order["size"],
                    "time_in_force": "post_only"
                })
                
        # Send batch order to CedeHub
        await self.cedehub._cedehub_request(
            "POST", "/v1/orders/batch", batch_request
        )
