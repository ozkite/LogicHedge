"""
CCXT Integration Layer for Logic Hedge
Unified trading interface for 100+ exchanges
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Exchange configuration for CCXT"""
    name: str
    api_key: str
    api_secret: str
    password: Optional[str] = None  # For some exchanges
    uid: Optional[str] = None  # For some exchanges
    sandbox: bool = True  # Start with testnet
    rate_limit: int = 10  # requests per second
    enable_rate_limit: bool = True
    
@dataclass
class UnifiedOrder:
    """Unified order format across all exchanges"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'limit', 'market', 'stop', etc.
    amount: float
    price: Optional[float] = None
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    average: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'canceled'
    fee: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    datetime: str = ''
    last_trade_timestamp: Optional[datetime] = None
    info: Dict = field(default_factory=dict)  # Raw exchange response
    
class CCXTUnifiedExchange:
    """
    Unified exchange interface using CCXT
    Supports 100+ exchanges with single API
    """
    
    def __init__(self, exchange_name: str, config: ExchangeConfig):
        self.exchange_name = exchange_name
        self.config = config
        
        # Initialize CCXT exchange
        exchange_class = getattr(ccxt_async, exchange_name.lower())
        
        exchange_params = {
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': config.enable_rate_limit,
            'rateLimit': config.rate_limit * 1000,  # Convert to milliseconds
        }
        
        if config.password:
            exchange_params['password'] = config.password
        if config.uid:
            exchange_params['uid'] = config.uid
            
        if config.sandbox:
            exchange_params['options'] = {'defaultType': 'spot'}
            # Enable testnet if supported
            if hasattr(exchange_class, 'sandbox'):
                exchange_params['sandbox'] = True
                
        self.exchange = exchange_class(exchange_params)
        
        # Exchange metadata
        self.symbols = []
        self.markets = {}
        self.balances = {}
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.utcnow()
        
    async def initialize(self):
        """Initialize exchange connection"""
        try:
            # Load markets
            await self.exchange.load_markets()
            self.symbols = list(self.exchange.symbols)
            self.markets = self.exchange.markets
            
            # Test connection
            await self.exchange.fetch_status()
            
            # Get initial balance
            await self.fetch_balance()
            
            logger.info(f"CCXT exchange initialized: {self.exchange_name}")
            logger.info(f"Available symbols: {len(self.symbols)}")
            logger.info(f"Testnet: {self.config.sandbox}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            return False
            
    async def fetch_balance(self, params: Dict = None) -> Dict:
        """Fetch account balance"""
        try:
            balance = await self.exchange.fetch_balance(params)
            self.balances = balance
            return balance
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            raise
            
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker for symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Ticker fetch error for {symbol}: {e}")
            raise
            
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Order book fetch error for {symbol}: {e}")
            raise
            
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None, 
                          params: Dict = None) -> UnifiedOrder:
        """Create a new order"""
        try:
            order = await self.exchange.create_order(
                symbol, order_type, side, amount, price, params
            )
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Order creation error: {e}")
            raise
            
    async def create_market_order(self, symbol: str, side: str, 
                                 amount: float) -> UnifiedOrder:
        """Create market order"""
        return await self.create_order(symbol, 'market', side, amount)
        
    async def create_limit_order(self, symbol: str, side: str, 
                                amount: float, price: float) -> UnifiedOrder:
        """Create limit order"""
        return await self.create_order(symbol, 'limit', side, amount, price)
        
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order"""
        try:
            # Some exchanges need symbol
            if symbol:
                result = await self.exchange.cancel_order(order_id, symbol)
            else:
                result = await self.exchange.cancel_order(order_id)
                
            return result.get('status') == 'canceled'
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            raise
            
    async def fetch_order(self, order_id: str, symbol: str = None) -> UnifiedOrder:
        """Fetch order by ID"""
        try:
            if symbol:
                order = await self.exchange.fetch_order(order_id, symbol)
            else:
                order = await self.exchange.fetch_order(order_id)
                
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Order fetch error: {e}")
            raise
            
    async def fetch_open_orders(self, symbol: str = None) -> List[UnifiedOrder]:
        """Fetch open orders"""
        try:
            if symbol:
                orders = await self.exchange.fetch_open_orders(symbol)
            else:
                orders = await self.exchange.fetch_open_orders()
                
            return [self._parse_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Open orders fetch error: {e}")
            raise
            
    async def fetch_closed_orders(self, symbol: str = None, 
                                 since: int = None, limit: int = 100) -> List[UnifiedOrder]:
        """Fetch closed orders"""
        try:
            orders = await self.exchange.fetch_closed_orders(
                symbol=symbol, since=since, limit=limit
            )
            return [self._parse_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Closed orders fetch error: {e}")
            raise
            
    async def fetch_my_trades(self, symbol: str = None, 
                             since: int = None, limit: int = 100) -> List[Dict]:
        """Fetch personal trades"""
        try:
            trades = await self.exchange.fetch_my_trades(
                symbol=symbol, since=since, limit=limit
            )
            return trades
        except Exception as e:
            logger.error(f"Trades fetch error: {e}")
            raise
            
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                         since: int = None, limit: int = 100) -> List[List]:
        """Fetch OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe, since, limit
            )
            return ohlcv
        except Exception as e:
            logger.error(f"OHLCV fetch error: {e}")
            raise
            
    def _parse_order(self, order: Dict) -> UnifiedOrder:
        """Parse CCXT order to unified format"""
        return UnifiedOrder(
            order_id=str(order.get('id')),
            symbol=order.get('symbol'),
            side=order.get('side'),
            type=order.get('type'),
            amount=float(order.get('amount', 0)),
            price=float(order.get('price', 0)) if order.get('price') else None,
            filled=float(order.get('filled', 0)),
            remaining=float(order.get('remaining', 0)),
            cost=float(order.get('cost', 0)),
            average=float(order.get('average', 0)) if order.get('average') else None,
            status=order.get('status', 'unknown'),
            fee=order.get('fee'),
            timestamp=datetime.fromtimestamp(order.get('timestamp', 0) / 1000),
            datetime=order.get('datetime', ''),
            last_trade_timestamp=datetime.fromtimestamp(
                order.get('lastTradeTimestamp', 0) / 1000
            ) if order.get('lastTradeTimestamp') else None,
            info=order.get('info', {})
        )
        
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate for perpetuals (if supported)"""
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding_rate = await self.exchange.fetch_funding_rate(symbol)
                return float(funding_rate.get('fundingRate', 0))
        except Exception as e:
            logger.debug(f"Funding rate not available for {self.exchange_name}: {e}")
        return None
        
    async def get_leverage(self, symbol: str) -> Optional[float]:
        """Get current leverage (for margin trading)"""
        try:
            if hasattr(self.exchange, 'fetch_leverage_tiers'):
                tiers = await self.exchange.fetch_leverage_tiers([symbol])
                if symbol in tiers:
                    return float(tiers[symbol][0].get('maxLeverage', 1))
        except Exception as e:
            logger.debug(f"Leverage not available for {self.exchange_name}: {e}")
        return None
        
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage (for margin trading)"""
        try:
            if hasattr(self.exchange, 'set_leverage'):
                await self.exchange.set_leverage(leverage, symbol)
                return True
        except Exception as e:
            logger.error(f"Set leverage error: {e}")
        return False
        
    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()
