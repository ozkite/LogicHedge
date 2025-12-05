"""
Layer 2 derivatives trading on dYdX and Aevo
Low fees, high speed, cross-margin
"""

class Layer2Derivatives:
    """
    Unified Layer 2 derivatives trading
    Combines dYdX (perpetuals) + Aevo (options)
    """
    
    def __init__(self, private_key: str, stark_key: str):
        self.private_key = private_key
        self.stark_key = stark_key
        
        # Initialize clients
        self.dydx_client = None
        self.aevo_client = None
        
    async def initialize(self):
        """Initialize Layer 2 connections"""
        # Initialize dYdX
        from dydx3 import Client
        from dydx3.constants import NETWORK_ID_MAINNET
        
        self.dydx_client = Client(
            network_id=NETWORK_ID_MAINNET,
            stark_private_key=self.stark_key,
            eth_private_key=self.private_key,
            default_ethereum_address=self._get_address()
        )
        
        # Initialize Aevo
        self.aevo_client = AevoClient(
            api_key=self._get_aevo_key(),
            api_secret=self._get_aevo_secret(),
            signing_key=self.private_key
        )
        
    async def execute_cross_margin_strategy(self):
        """
        Cross-margin strategy using dYdX
        Use winning positions as collateral for new trades
        """
        # Get account info
        account = self.dydx_client.private.get_account()
        free_collateral = float(account['account']['freeCollateral'])
        
        # Calculate available margin
        available_margin = free_collateral * 5  # 5x leverage
        
        # Find best opportunity
        opportunities = await self._scan_layer2_opportunities()
        
        for opp in opportunities:
            if opp['expected_return'] > 0.01:  # 1% min
                # Execute with cross-margin
                order = await self.dydx_client.private.create_order(
                    position_id=opp['position_id'],
                    market=opp['market'],
                    side=opp['side'],
                    order_type='MARKET',
                    size=opp['size'],
                    price=None,
                    leverage=opp.get('leverage', 5)
                )
                
                logger.info(f"Cross-margin trade: {opp['market']} {opp['side']} "
                          f"size={opp['size']}, leverage={opp.get('leverage', 5)}x")
                          
    async def execute_options_strategy(self, strategy: str, params: Dict):
        """
        Execute options strategy on Aevo
        """
        if strategy == "covered_call":
            # Sell call against spot position
            await self._execute_covered_call(params)
            
        elif strategy == "cash_secured_put":
            # Sell put with cash collateral
            await self._execute_cash_secured_put(params)
            
        elif strategy == "iron_condor":
            # Sell OTM call + put spread
            await self._execute_iron_condor(params)
            
    async def execute_perp_options_arbitrage(self):
        """
        Arbitrage between perpetuals (dYdX) and options (Aevo)
        """
        # Get perpetual funding rate
        perp_markets = self.dydx_client.public.get_markets()
        btc_perp = perp_markets['markets']['BTC-USD']
        funding_rate = float(btc_perp['nextFundingRate'])
        
        # Get options implied volatility
        options_chain = await self.aevo_client.get_options_chain('BTC')
        atm_iv = options_chain['atm_iv']
        
        # Calculate arbitrage opportunity
        if funding_rate > 0.0005 and atm_iv < 0.5:  # Specific conditions
            # Sell perp (receive funding), buy options (cheap vol)
            await self._execute_vol_carry_trade(funding_rate, atm_iv)
            
    async def execute_low_latency_arbitrage(self):
        """
        Low-latency arbitrage between Layer 1 and Layer 2
        """
        # Monitor price differences
        while True:
            # Get L1 price (Coinbase)
            l1_price = await self._get_l1_price('BTC')
            
            # Get L2 price (dYdX)
            l2_price = await self._get_l2_price('BTC-USD')
            
            # Calculate spread
            spread = abs(l1_price - l2_price) / min(l1_price, l2_price)
            
            if spread > 0.001:  # 0.1% opportunity
                if l1_price > l2_price:
                    # Buy on dYdX, sell on Coinbase
                    await self._execute_cross_layer_arb('buy', 'dydx', 'sell', 'coinbase')
                else:
                    # Buy on Coinbase, sell on dYdX
                    await self._execute_cross_layer_arb('buy', 'coinbase', 'sell', 'dydx')
                    
            await asyncio.sleep(0.01)  # 10ms scan
