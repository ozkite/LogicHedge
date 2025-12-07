"""
Trading strategies built on top of CCXT
"""

import asyncio
from typing import Dict, List, Optional, Any
import talib
import numpy as np
from datetime import datetime, timedelta

class CCXTTradingStrategies:
    """
    Collection of trading strategies using CCXT
    """
    
    def __init__(self, exchange_manager: MultiExchangeManager):
        self.manager = exchange_manager
        self.strategies = {}
        
    async def run_mean_reversion(self, symbol: str, exchange_name: str, 
                                window: int = 20, std_dev: float = 2.0):
        """
        Mean reversion strategy using Bollinger Bands
        Buy when price crosses below lower band, sell when above upper band
        """
        exchange = self.manager.exchanges.get(exchange_name)
        if not exchange:
            return
            
        while True:
            try:
                # Fetch OHLCV data
                ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=window + 10)
                closes = np.array([candle[4] for candle in ohlcv])
                
                if len(closes) >= window:
                    # Calculate Bollinger Bands
                    upper, middle, lower = talib.BBANDS(
                        closes, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev
                    )
                    
                    current_price = closes[-1]
                    
                    # Trading signals
                    if current_price < lower[-1]:
                        # Price below lower band - buy signal
                        balance = await exchange.fetch_balance()
                        quote_currency = symbol.split('/')[1]
                        available = balance.get(quote_currency, {}).get('free', 0)
                        
                        if available > 10:  # Minimum $10
                            amount = (available * 0.1) / current_price  # Use 10% of balance
                            await exchange.create_market_order(symbol, 'buy', amount)
                            logger.info(f"Mean reversion BUY: {symbol} @ {current_price}")
                            
                    elif current_price > upper[-1]:
                        # Price above upper band - sell signal
                        balance = await exchange.fetch_balance()
                        base_currency = symbol.split('/')[0]
                        available = balance.get(base_currency, {}).get('free', 0)
                        
                        if available > 0:
                            await exchange.create_market_order(symbol, 'sell', available)
                            logger.info(f"Mean reversion SELL: {symbol} @ {current_price}")
                            
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Mean reversion error: {e}")
                await asyncio.sleep(60)
                
    async def run_momentum_strategy(self, symbol: str, exchange_name: str,
                                  fast_period: int = 12, slow_period: int = 26):
        """
        Momentum strategy using MACD
        Buy when MACD crosses above signal, sell when below
        """
        exchange = self.manager.exchanges.get(exchange_name)
        if not exchange:
            return
            
        while True:
            try:
                # Fetch OHLCV data
                ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=slow_period + 10)
                closes = np.array([candle[4] for candle in ohlcv])
                
                if len(closes) >= slow_period:
                    # Calculate MACD
                    macd, signal, hist = talib.MACD(
                        closes, fastperiod=fast_period, 
                        slowperiod=slow_period, signalperiod=9
                    )
                    
                    # Trading signals
                    if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
                        # MACD crossed above signal - buy signal
                        balance = await exchange.fetch_balance()
                        quote_currency = symbol.split('/')[1]
                        available = balance.get(quote_currency, {}).get('free', 0)
                        
                        if available > 10:
                            amount = (available * 0.1) / closes[-1]
                            await exchange.create_market_order(symbol, 'buy', amount)
                            logger.info(f"Momentum BUY: {symbol} - MACD crossover")
                            
                    elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
                        # MACD crossed below signal - sell signal
                        balance = await exchange.fetch_balance()
                        base_currency = symbol.split('/')[0]
                        available = balance.get(base_currency, {}).get('free', 0)
                        
                        if available > 0:
                            await exchange.create_market_order(symbol, 'sell', available)
                            logger.info(f"Momentum SELL: {symbol} - MACD crossunder")
                            
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Momentum strategy error: {e}")
                await asyncio.sleep(60)
                
    async run_grid_trading(self, symbol: str, exchange_name: str,
                          lower_price: float, upper_price: float,
                          grid_number: int = 10, investment: float = 1000):
        """
        Grid trading strategy
        Place buy orders below current price, sell orders above
        """
        exchange = self.manager.exchanges.get(exchange_name)
        if not exchange:
            return
            
        # Calculate grid levels
        price_step = (upper_price - lower_price) / grid_number
        grid_levels = [lower_price + i * price_step for i in range(grid_number + 1)]
        
        # Initial orders
        orders = []
        
        # Get current price
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Place initial grid orders
        for i, level in enumerate(grid_levels):
            if level < current_price:
                # Buy order
                order = await exchange.create_limit_order(
                    symbol, 'buy', investment / grid_number / level, level
                )
                orders.append(order)
            elif level > current_price:
                # Sell order (need to have the asset first)
                # For now, just track levels
                pass
                
        # Monitor and rebalance
        while True:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Check if any orders were filled
                open_orders = await exchange.fetch_open_orders(symbol)
                order_ids = [order.order_id for order in open_orders]
                
                # Re-fill grid if orders were filled
                for i, level in enumerate(grid_levels):
                    if level < current_price:
                        # Check if we have a buy order at this level
                        has_order = any(
                            abs(float(order.price or 0) - level) < price_step * 0.1
                            for order in open_orders if order.side == 'buy'
                        )
                        
                        if not has_order:
                            # Place new buy order
                            order = await exchange.create_limit_order(
                                symbol, 'buy', investment / grid_number / level, level
                            )
                            orders.append(order)
                            
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Grid trading error: {e}")
                await asyncio.sleep(300)
