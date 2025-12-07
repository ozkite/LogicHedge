"""
Multi-exchange manager for CCXT exchanges
Orchestrates trading across multiple exchanges simultaneously
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class MultiExchangeManager:
    """
    Manages multiple CCXT exchanges with unified interface
    """
    
    def __init__(self, configs: Dict[str, ExchangeConfig]):
        self.configs = configs
        self.exchanges: Dict[str, CCXTUnifiedExchange] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Price synchronization
        self.price_cache = {}
        self.orderbook_cache = {}
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0,
            'total_fees': 0.0
        }
        
    async def initialize_all(self):
        """Initialize all exchanges"""
        tasks = []
        for name, config in self.configs.items():
            exchange = CCXTUnifiedExchange(name, config)
            self.exchanges[name] = exchange
            tasks.append(exchange.initialize())
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log initialization results
        for name, result in zip(self.configs.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize {name}: {result}")
            else:
                logger.info(f"Initialized {name}: {result}")
                
        # Start price synchronization
        asyncio.create_task(self._sync_prices())
        
    async def _sync_prices(self):
        """Synchronize prices across all exchanges"""
        while True:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    # Update prices for major symbols
                    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'][:3]  # Top 3
                    
                    for symbol in symbols:
                        try:
                            ticker = await exchange.fetch_ticker(symbol)
                            if exchange_name not in self.price_cache:
                                self.price_cache[exchange_name] = {}
                            self.price_cache[exchange_name][symbol] = ticker
                            
                            # Update orderbook
                            orderbook = await exchange.fetch_order_book(symbol, 10)
                            if exchange_name not in self.orderbook_cache:
                                self.orderbook_cache[exchange_name] = {}
                            self.orderbook_cache[exchange_name][symbol] = orderbook
                            
                        except Exception as e:
                            logger.debug(f"Price sync error for {exchange_name}/{symbol}: {e}")
                            
                await asyncio.sleep(5)  # Sync every 5 seconds
                
            except Exception as e:
                logger.error(f"Price synchronization error: {e}")
                await asyncio.sleep(30)
                
    async def find_best_price(self, symbol: str, side: str) -> Tuple[str, float]:
        """
        Find best price across all exchanges
        Returns: (exchange_name, price)
        """
        best_price = None
        best_exchange = None
        
        for exchange_name, prices in self.price_cache.items():
            if symbol in prices:
                ticker = prices[symbol]
                price = ticker['ask'] if side == 'buy' else ticker['bid']
                
                if best_price is None:
                    best_price = price
                    best_exchange = exchange_name
                elif side == 'buy' and price < best_price:
                    best_price = price
                    best_exchange = exchange_name
                elif side == 'sell' and price > best_price:
                    best_price = price
                    best_exchange = exchange_name
                    
        return best_exchange, best_price
        
    async def execute_smart_order(self, symbol: str, side: str, amount: float, 
                                 order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute order on best exchange with smart routing
        """
        # Find best exchange
        best_exchange, best_price = await self.find_best_price(symbol, side)
        
        if not best_exchange:
            raise ValueError(f"No exchange found for {symbol}")
            
        exchange = self.exchanges[best_exchange]
        
        # Check balance
        balance = await exchange.fetch_balance()
        quote_currency = symbol.split('/')[1]
        
        if side == 'buy':
            required = amount * best_price * 1.01  # 1% buffer
            available = balance.get(quote_currency, {}).get('free', 0)
            if available < required:
                raise ValueError(f"Insufficient {quote_currency} balance: "
                               f"available={available}, required={required}")
        else:
            base_currency = symbol.split('/')[0]
            available = balance.get(base_currency, {}).get('free', 0)
            if available < amount:
                raise ValueError(f"Insufficient {base_currency} balance: "
                               f"available={available}, required={amount}")
                               
        # Execute order
        if order_type == 'market':
            order = await exchange.create_market_order(symbol, side, amount)
        elif order_type == 'limit':
            order = await exchange.create_limit_order(symbol, side, amount, best_price)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
            
        # Update metrics
        self.metrics['total_trades'] += 1
        if order.status == 'closed':
            self.metrics['successful_trades'] += 1
            self.metrics['total_volume'] += order.cost
            if order.fee:
                self.metrics['total_fees'] += order.fee.get('cost', 0)
        else:
            self.metrics['failed_trades'] += 1
            
        return {
            'exchange': best_exchange,
            'order': order,
            'best_price': best_price
        }
        
    async def execute_cross_exchange_arbitrage(self, symbol: str, 
                                              min_profit_pct: float = 0.1) -> Optional[Dict]:
        """
        Execute cross-exchange arbitrage
        Buy on exchange with lowest price, sell on exchange with highest price
        """
        # Get all prices
        prices = {}
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                prices[exchange_name] = {
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last']
                }
            except Exception as e:
                logger.debug(f"Price fetch error for {exchange_name}: {e}")
                
        if len(prices) < 2:
            return None
            
        # Find lowest ask and highest bid
        lowest_ask_exchange = min(prices.items(), key=lambda x: x[1]['ask'])
        highest_bid_exchange = max(prices.items(), key=lambda x: x[1]['bid'])
        
        buy_price = lowest_ask_exchange[1]['ask']
        sell_price = highest_bid_exchange[1]['bid']
        
        # Calculate profit
        profit_pct = ((sell_price - buy_price) / buy_price) * 100
        
        if profit_pct > min_profit_pct:
            # Check if we have enough balance on both exchanges
            buy_exchange = self.exchanges[lowest_ask_exchange[0]]
            sell_exchange = self.exchanges[highest_bid_exchange[0]]
            
            # Calculate maximum amount we can trade
            buy_balance = await buy_exchange.fetch_balance()
            sell_balance = await sell_exchange.fetch_balance()
            
            quote_currency = symbol.split('/')[1]
            base_currency = symbol.split('/')[0]
            
            # Maximum based on balances
            max_from_buy = buy_balance.get(quote_currency, {}).get('free', 0) / buy_price
            max_from_sell = sell_balance.get(base_currency, {}).get('free', 0)
            
            trade_amount = min(max_from_buy, max_from_sell, 1.0)  # Max 1 unit for safety
            
            if trade_amount > 0:
                # Execute simultaneously
                buy_task = asyncio.create_task(
                    buy_exchange.create_market_order(symbol, 'buy', trade_amount)
                )
                sell_task = asyncio.create_task(
                    sell_exchange.create_market_order(symbol, 'sell', trade_amount)
                )
                
                buy_result, sell_result = await asyncio.gather(buy_task, sell_task)
                
                profit = (sell_price - buy_price) * trade_amount
                
                return {
                    'buy_exchange': lowest_ask_exchange[0],
                    'sell_exchange': highest_bid_exchange[0],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'amount': trade_amount,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'buy_order': buy_result,
                    'sell_order': sell_result
                }
                
        return None
        
    async def get_unified_balance(self) -> Dict[str, float]:
        """Get unified balance across all exchanges"""
        unified_balance = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                for currency, data in balance['total'].items():
                    if currency not in unified_balance:
                        unified_balance[currency] = 0.0
                    unified_balance[currency] += float(data)
            except Exception as e:
                logger.error(f"Balance fetch error for {exchange_name}: {e}")
                
        return unified_balance
        
    async def close_all(self):
        """Close all exchange connections"""
        tasks = []
        for exchange in self.exchanges.values():
            tasks.append(exchange.close())
            
        await asyncio.gather(*tasks, return_exceptions=True)
