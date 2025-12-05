"""
FalconX institutional execution with best price routing
"""

class FalconXExecution:
    """
    FalconX institutional execution platform
    Best price routing across 50+ liquidity venues
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.falconx.io"
        
        # Execution venues
        self.venues = []
        self.liquidity_sources = []
        
    async def get_best_execution(self, symbol: str, side: str, size: float) -> Dict:
        """
        Get best execution price across all venues
        """
        quote = await self._falconx_request(
            "POST", "/v1/quote",
            {
                "client_order_id": f"fx_{int(time.time())}",
                "symbol": symbol,
                "side": side.upper(),
                "quantity": str(size),
                "quantity_currency": symbol.split('/')[1],
                "venue": "all",  # Search all venues
                "request_type": "indicative"
            }
        )
        
        # Analyze execution options
        best_execution = self._analyze_execution_options(quote)
        
        return best_execution
        
    async def execute_block_trade(self, trade_request: Dict) -> Dict:
        """
        Execute large block trade with minimal market impact
        """
        # Request block trade
        block_trade = await self._falconx_request(
            "POST", "/v1/block-trade",
            {
                "client_order_id": trade_request['client_order_id'],
                "symbol": trade_request['symbol'],
                "side": trade_request['side'],
                "quantity": trade_request['quantity'],
                "price": trade_request.get('price'),  # Optional for limit
                "time_in_force": trade_request.get('time_in_force', 'DAY'),
                "venue_preference": trade_request.get('venue_preference', []),
                "minimum_fill": trade_request.get('minimum_fill', 0.8)
            }
        )
        
        # FalconX will work the order across venues
        logger.info(f"Block trade initiated: {block_trade['block_trade_id']}, "
                  f"working {trade_request['quantity']} {trade_request['symbol']}")
                  
        return block_trade
        
    async def execute_twap_vwap(self, order_request: Dict) -> Dict:
        """
        Execute Time-Weighted or Volume-Weighted Average Price
        """
        algo_order = await self._falconx_request(
            "POST", "/v1/algo-order",
            {
                "client_order_id": order_request['client_order_id'],
                "symbol": order_request['symbol'],
                "side": order_request['side'],
                "quantity": order_request['quantity'],
                "algorithm": order_request['algorithm'],  # TWAP, VWAP, Iceberg
                "parameters": order_request.get('parameters', {}),
                "start_time": order_request.get('start_time'),
                "end_time": order_request.get('end_time')
            }
        )
        
        logger.info(f"Algo order started: {algo_order['algo_order_id']}, "
                  f"{order_request['algorithm']} for {order_request['quantity']}")
                  
        return algo_order
        
    def _analyze_execution_options(self, quote: Dict) -> Dict:
        """Analyze best execution venue"""
        executions = quote.get('execution_options', [])
        
        if not executions:
            return {}
            
        # Score each execution option
        scored = []
        for ex in executions:
            score = self._calculate_execution_score(ex)
            scored.append((score, ex))
            
        # Get best execution
        best_score, best_execution = max(scored, key=lambda x: x[0])
        
        return {
            'venue': best_execution['venue'],
            'price': best_execution['price'],
            'estimated_fill': best_execution['estimated_fill'],
            'estimated_slippage': best_execution['estimated_slippage'],
            'fee': best_execution['fee'],
            'score': best_score
        }
        
    def _calculate_execution_score(self, execution: Dict) -> float:
        """Calculate execution quality score"""
        score = 0.0
        
        # Price (higher is better for sell, lower for buy)
        price_score = 1.0 / (1.0 + abs(execution.get('price_improvement', 0)))
        score += price_score * 0.4
        
        # Fill probability
        fill_score = execution.get('fill_probability', 0.5)
        score += fill_score * 0.3
        
        # Fees (lower is better)
        fee_score = 1.0 / (1.0 + execution.get('fee_bps', 10))
        score += fee_score * 0.2
        
        # Speed
        speed_score = 1.0 / (1.0 + execution.get('latency_ms', 100) / 1000)
        score += speed_score * 0.1
        
        return score
