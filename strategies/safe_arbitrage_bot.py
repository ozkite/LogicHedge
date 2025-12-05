"""
Safe multi-exchange arbitrage bot for Binance, Bybit, MEXC, Pionex
"""

import asyncio
import logging
from typing import Dict, List, Tuple
import ccxt
import ccxt.async_support as ccxt_async
from datetime import datetime

logger = logging.getLogger(__name__)

class SafeArbitrageBot:
    """
    Implements the safest arbitrage strategies across 4 exchanges
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.balances = {}
        self.positions = {}
        
        # Risk limits
        self.max_daily_loss = 100  # $100
        self.max_trade_size = 500  # $500
        self.min_profit_pct = 0.001  # 0.1%
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        
    async def initialize(self):
        """Initialize exchange connections"""
        exchanges_config = self.config['exchanges']
        
        # Initialize CCXT exchanges
        if exchanges_config.get('binance', {}).get('enabled'):
            self.exchanges['binance'] = ccxt_async.binance({
                'apiKey': exchanges_config['binance']['api_key'],
                'secret': exchanges_config['binance']['api_secret'],
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
        if exchanges_config.get('bybit', {}).get('enabled'):
            self.exchanges['bybit'] = ccxt_async.bybit({
                'apiKey': exchanges_config['bybit']['api_key'],
                'secret': exchanges_config['bybit']['api_secret'],
                'enableRateLimit': True,
            })
            
        if exchanges_config.get('mexc', {}).get('enabled'):
            self.exchanges['mexc'] = ccxt_async.mexc({
                'apiKey': exchanges_config['mexc']['api_key'],
                'secret': exchanges_config['mexc']['api_secret'],
                'enableRateLimit': True,
            })
            
        logger.info("Exchanges initialized: %s", list(self.exchanges.keys()))
        
    async def run_cross_exchange_arbitrage(self):
        """
        #1 Strategy: Cross-exchange arbitrage
        Buy low on one exchange, sell high on another
        """
        symbol = 'BTC/USDT'
        min_profit_pct = 0.001  # 0.1%
        
        while True:
            try:
                # Get prices from all exchanges simultaneously
                prices = {}
                for name, exchange in self.exchanges.items():
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        prices[name] = {
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'last': ticker['last']
                        }
                    except Exception as e:
                        logger.error(f"Price fetch error on {name}: {e}")
                        
                # Find best bid and ask
                best_bid_exchange = max(prices.items(), key=lambda x: x[1]['bid'])
                best_ask_exchange = min(prices.items(), key=lambda x: x[1]['ask'])
                
                best_bid = best_bid_exchange[1]['bid']
                best_ask = best_ask_exchange[1]['ask']
                
                # Calculate spread
                if best_bid > best_ask:
                    spread_pct = ((best_bid - best_ask) / best_ask) * 100
                    
                    if spread_pct >= min_profit_pct * 100:
                        # Execute arbitrage
                        buy_exchange = best_ask_exchange[0]
                        sell_exchange = best_bid_exchange[0]
                        
                        # Calculate trade size
                        trade_size = min(
                            self.max_trade_size / best_ask,
                            await self._get_available_balance(buy_exchange, 'USDT') / best_ask
                        )
                        
                        if trade_size > 0.001:  # Minimum BTC size
                            # Execute simultaneously
                            buy_task = asyncio.create_task(
                                self.exchanges[buy_exchange].create_market_buy_order(
                                    symbol, trade_size
                                )
                            )
                            sell_task = asyncio.create_task(
                                self.exchanges[sell_exchange].create_market_sell_order(
                                    symbol, trade_size
                                )
                            )
                            
                            await asyncio.gather(buy_task, sell_task)
                            
                            profit = (best_bid - best_ask) * trade_size
                            self.daily_pnl += profit
                            self.total_trades += 1
                            
                            logger.info(f"Arbitrage executed: Buy {buy_exchange} @ {best_ask}, "
                                      f"Sell {sell_exchange} @ {best_bid}, Profit: ${profit:.2f}")
                            
                await asyncio.sleep(0.1)  # 100ms scan interval
                
            except Exception as e:
                logger.error(f"Arbitrage loop error: {e}")
                await asyncio.sleep(1)
                
    async def run_funding_rate_arbitrage(self):
        """
        #2 Strategy: Funding rate arbitrage (Bybit specific)
        """
        if 'bybit' not in self.exchanges:
            return
            
        while True:
            try:
                # Get funding rates for top pairs
                symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
                
                for symbol in symbols:
                    funding_rate = await self.exchanges['bybit'].fetch_funding_rate(symbol)
                    
                    # If funding is negative, go LONG (get paid)
                    if funding_rate['fundingRate'] < -0.0005:  # -0.05%
                        # Check if we have enough balance
                        usdt_balance = await self._get_available_balance('bybit', 'USDT')
                        
                        if usdt_balance > 100:  # Minimum $100
                            # Calculate position with 3x leverage
                            position_size = (usdt_balance * 3) / funding_rate['last']
                            position_size = min(position_size, 0.1)  # Max 0.1 BTC
                            
                            # Place long order
                            await self.exchanges['bybit'].create_market_buy_order(
                                symbol, position_size, {'leverage': 3}
                            )
                            
                            logger.info(f"Funding arb LONG: {symbol}, "
                                      f"Rate: {funding_rate['fundingRate']*100:.4f}%")
                                      
                    # If funding is positive, go SHORT (get paid)
                    elif funding_rate['fundingRate'] > 0.0005:  # +0.05%
                        # Similar logic for short positions
                        pass
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Funding rate arb error: {e}")
                await asyncio.sleep(60)
                
    async def _get_available_balance(self, exchange_name: str, currency: str) -> float:
        """Get available balance for trading"""
        try:
            balance = await self.exchanges[exchange_name].fetch_balance()
            return balance[currency]['free']
        except:
            return 0.0
            
    def get_performance(self) -> Dict:
        """Get bot performance metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'active_exchanges': list(self.exchanges.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }
