"""
Bridge between CCXT exchanges and LogicHedge systems
"""

class CCXTBridge:
    """
    Bridges CCXT exchanges to LogicHedge trading systems
    """
    
    def __init__(self, exchange_manager: MultiExchangeManager):
        self.manager = exchange_manager
        
    async def get_market_data_for_research(self) -> Dict[str, Any]:
        """
        Get comprehensive market data for research module
        """
        market_data = {
            'prices': {},
            'orderbooks': {},
            'volumes': {},
            'funding_rates': {},
            'timestamps': {}
        }
        
        for exchange_name, exchange in self.manager.exchanges.items():
            market_data['prices'][exchange_name] = {}
            market_data['orderbooks'][exchange_name] = {}
            market_data['volumes'][exchange_name] = {}
            market_data['funding_rates'][exchange_name] = {}
            
            # Get top symbols
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            
            for symbol in symbols:
                try:
                    # Get ticker
                    ticker = await exchange.fetch_ticker(symbol)
                    market_data['prices'][exchange_name][symbol] = {
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'last': ticker['last'],
                        'volume': ticker['quoteVolume']
                    }
                    
                    # Get orderbook
                    orderbook = await exchange.fetch_order_book(symbol, 10)
                    market_data['orderbooks'][exchange_name][symbol] = {
                        'bids': orderbook['bids'][:5],
                        'asks': orderbook['asks'][:5]
                    }
                    
                    # Get funding rate if available
                    funding_rate = await exchange.get_funding_rate(symbol)
                    if funding_rate:
                        market_data['funding_rates'][exchange_name][symbol] = funding_rate
                        
                except Exception as e:
                    logger.debug(f"Market data error for {exchange_name}/{symbol}: {e}")
                    
        market_data['timestamps']['fetched_at'] = datetime.utcnow().isoformat()
        
        return market_data
        
    async def execute_logichedge_strategy(self, strategy_signal: Dict):
        """
        Execute LogicHedge strategy signal through CCXT
        """
        symbol = strategy_signal.get('symbol')
        side = strategy_signal.get('side')
        amount = strategy_signal.get('amount')
        order_type = strategy_signal.get('order_type', 'market')
        
        if not all([symbol, side, amount]):
            raise ValueError("Missing required strategy signal parameters")
            
        # Use smart order routing
        result = await self.manager.execute_smart_order(
            symbol, side, amount, order_type
        )
        
        return result
        
    async def get_portfolio_for_risk_manager(self) -> Dict[str, Any]:
        """
        Get unified portfolio for risk management
        """
        portfolio = {
            'balances': {},
            'positions': {},
            'exposures': {},
            'total_value': 0.0
        }
        
        # Get balances from all exchanges
        for exchange_name, exchange in self.manager.exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                portfolio['balances'][exchange_name] = balance
                
                # Calculate total value
                for currency, data in balance['total'].items():
                    value = float(data)
                    if value > 0:
                        # Get price for valuation
                        try:
                            if currency != 'USDT':
                                symbol = f"{currency}/USDT"
                                ticker = await exchange.fetch_ticker(symbol)
                                value_usd = value * ticker['last']
                            else:
                                value_usd = value
                                
                            portfolio['total_value'] += value_usd
                        except:
                            # If can't price, skip
                            pass
                            
                # Get open positions (for derivatives exchanges)
                if hasattr(exchange.exchange, 'fetch_positions'):
                    positions = await exchange.exchange.fetch_positions()
                    portfolio['positions'][exchange_name] = positions
                    
            except Exception as e:
                logger.error(f"Portfolio fetch error for {exchange_name}: {e}")
                
        return portfolio
