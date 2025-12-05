"""
CedeHub-powered strategies leveraging unified margin and cross-collateral
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from web3 import Web3
import hashlib
import hmac
import time

logger = logging.getLogger(__name__)

@dataclass
class CedeHubPosition:
    """Cross-exchange position with unified margin"""
    position_id: str
    asset: str
    total_size: float
    exchange_allocations: Dict[str, float]  # {exchange: size}
    collateral_asset: str = "USDC"
    collateral_amount: float = 0.0
    leverage: float = 1.0
    unified_margin_ratio: float = 0.0
    
class CedeHubArbitrage:
    """
    Advanced arbitrage using CedeHub's unified margin
    Execute larger positions without moving funds
    """
    
    def __init__(self, cedehub_api_key: str, cedehub_secret: str):
        self.api_key = cedehub_api_key
        self.secret = cedehub_secret
        self.base_url = "https://api.cedehub.io"
        
        # Connected exchanges via CedeHub
        self.connected_exchanges = []
        self.unified_balance = 0.0
        self.available_margin = 0.0
        
    async def initialize(self):
        """Initialize CedeHub connection"""
        # Get connected exchanges
        exchanges = await self._cedehub_request("GET", "/v1/exchanges")
        self.connected_exchanges = [ex['name'] for ex in exchanges]
        
        # Get unified balance
        balance = await self._cedehub_request("GET", "/v1/balance")
        self.unified_balance = float(balance['total_collateral_value'])
        self.available_margin = float(balance['available_margin'])
        
        logger.info(f"CedeHub initialized: ${self.unified_balance:,.2f} collateral, "
                  f"Exchanges: {self.connected_exchanges}")
                  
    async def execute_unified_arbitrage(self):
        """
        Advanced arbitrage using CedeHub's unified margin
        No fund transfers needed between exchanges
        """
        # 1. Find price discrepancies across ALL connected exchanges
        price_matrix = await self._build_price_matrix()
        
        # 2. Find profitable arbitrage paths
        opportunities = self._find_arbitrage_paths(price_matrix)
        
        # 3. Execute using CedeHub's smart order routing
        for opp in opportunities:
            if opp['expected_return_pct'] > 0.002:  # 0.2% min
                await self._execute_unified_trade(opp)
                
    async def _build_price_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build price matrix across all exchanges"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        matrix = {}
        
        for symbol in symbols:
            matrix[symbol] = {}
            
            # Get prices from all exchanges via CedeHub
            prices = await self._cedehub_request(
                "GET", f"/v1/prices/{symbol.replace('/', '')}"
            )
            
            for exchange_data in prices['exchanges']:
                exchange = exchange_data['exchange']
                bid = float(exchange_data['bid'])
                ask = float(exchange_data['ask'])
                matrix[symbol][exchange] = {'bid': bid, 'ask': ask}
                
        return matrix
        
    def _find_arbitrage_paths(self, matrix: Dict) -> List[Dict]:
        """Find triangular and cross-exchange arbitrage paths"""
        opportunities = []
        
        # Example: BTC/USDT price differences
        btc_prices = matrix['BTC/USDT']
        exchanges = list(btc_prices.keys())
        
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                # Check direct arbitrage
                spread = btc_prices[ex2]['bid'] - btc_prices[ex1]['ask']
                spread_pct = (spread / btc_prices[ex1]['ask']) * 100
                
                if spread_pct > 0.1:  # 0.1% opportunity
                    opportunities.append({
                        'type': 'direct',
                        'buy_exchange': ex1,
                        'sell_exchange': ex2,
                        'symbol': 'BTC/USDT',
                        'spread_pct': spread_pct,
                        'expected_return_pct': spread_pct - 0.05,  # minus fees
                        'max_size': self._calculate_max_size(ex1, ex2, 'BTC/USDT')
                    })
                    
        return opportunities
        
    async def _execute_unified_trade(self, opportunity: Dict):
        """Execute trade using CedeHub's unified execution"""
        trade_request = {
            "type": "arbitrage",
            "strategy": "cross_exchange",
            "legs": [
                {
                    "exchange": opportunity['buy_exchange'],
                    "symbol": opportunity['symbol'],
                    "side": "buy",
                    "type": "market",
                    "size": opportunity['max_size']
                },
                {
                    "exchange": opportunity['sell_exchange'],
                    "symbol": opportunity['symbol'],
                    "side": "sell",
                    "type": "market",
                    "size": opportunity['max_size']
                }
            ],
            "collateral_asset": "USDC",
            "leverage": 1.0,
            "slippage_tolerance": 0.001
        }
        
        # Submit to CedeHub for unified execution
        result = await self._cedehub_request(
            "POST", "/v1/trade/unified", trade_request
        )
        
        logger.info(f"CedeHub unified trade executed: {result['trade_id']}, "
                  f"Expected profit: {opportunity['expected_return_pct']:.3f}%")
                  
    async def execute_cross_collateral_strategy(self):
        """
        Use BTC as collateral to trade ETH/SOL/etc.
        Avoid selling BTC (no tax implications)
        """
        # 1. Deposit BTC as collateral
        deposit_tx = await self._cedehub_request(
            "POST", "/v1/collateral/deposit",
            {"asset": "BTC", "amount": 0.1}  # Deposit 0.1 BTC
        )
        
        # 2. Get borrowing power
        portfolio = await self._cedehub_request("GET", "/v1/portfolio")
        btc_collateral = float(portfolio['collateral']['BTC']['value'])
        borrowing_power = btc_collateral * 3  # 3x leverage
        
        # 3. Use borrowed USDC to trade alts
        trade_size = borrowing_power * 0.3  # Use 30% of borrowing power
        
        # Buy ETH with borrowed USDC
        trade = {
            "exchange": "binance",
            "symbol": "ETH/USDT",
            "side": "buy",
            "type": "market",
            "size": trade_size / 3000,  # ETH price ~$3000
            "collateral_asset": "BTC",
            "is_cross_collateral": True
        }
        
        await self._cedehub_request("POST", "/v1/trade/execute", trade)
        
        logger.info(f"Cross-collateral trade: Used 0.1 BTC to trade ${trade_size:,.2f} of ETH")
        
    async def execute_portfolio_hedging(self):
        """
        Hedge entire portfolio across multiple exchanges
        Use derivatives on one exchange to hedge spot on another
        """
        # Get portfolio delta across all exchanges
        portfolio = await self._cedehub_request("GET", "/v1/portfolio/risk")
        net_delta = float(portfolio['greek_exposures']['delta'])
        
        if abs(net_delta) > 10000:  # More than $10k delta exposure
            # Calculate hedge size
            hedge_size = -net_delta  # Opposite direction
            
            # Execute hedge using perpetuals on Bybit
            hedge_trade = {
                "exchange": "bybit",
                "symbol": "BTC/USDT:USDT",
                "side": "sell" if net_delta > 0 else "buy",
                "type": "market",
                "size": abs(hedge_size) / 50000,  # BTC price
                "is_hedge": True
            }
            
            await self._cedehub_request("POST", "/v1/trade/hedge", hedge_trade)
            
            logger.info(f"Portfolio hedged: ${hedge_size:,.2f} delta, "
                      f"using {hedge_trade['exchange']} perpetuals")
                      
    async def execute_smart_order_routing(self, symbol: str, side: str, size: float):
        """
        Use CedeHub's smart order routing for best execution
        Automatically splits order across multiple exchanges
        """
        sor_request = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "strategy": "twap",  # or "vwap", "market", "iceberg"
            "duration_seconds": 30,
            "max_slippage": 0.001,
            "min_fill": 0.95
        }
        
        # CedeHub will route to best exchanges
        result = await self._cedehub_request(
            "POST", "/v1/execute/smart-router", sor_request
        )
        
        logger.info(f"Smart order routed: {size} {symbol} {side}, "
                  f"avg price: {result['avg_price']}, "
                  f"exchanges used: {result['executions']}")
                  
    async def _cedehub_request(self, method: str, endpoint: str, data: Dict = None):
        """Make authenticated request to CedeHub API"""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        # Prepare signature
        message = timestamp + method + endpoint
        if data:
            message += json.dumps(data)
            
        signature = hmac.new(
            self.secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()
