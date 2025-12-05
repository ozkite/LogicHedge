"""
Unified portfolio management across Binance, Bybit, MEXC via CedeHub
"""

class CedeHubPortfolioManager:
    """
    Manage portfolio across multiple exchanges with:
    - Unified P&L tracking
    - Cross-exchange rebalancing
    - Risk aggregation
    - Tax optimization
    """
    
    def __init__(self, cedehub_client):
        self.cedehub = cedehub_client
        self.portfolio = {}
        self.risk_metrics = {}
        
    async def get_unified_portfolio(self) -> Dict:
        """Get unified portfolio across all exchanges"""
        portfolio = await self.cedehub._cedehub_request("GET", "/v1/portfolio")
        
        # Aggregate positions
        unified = {}
        for exchange, positions in portfolio['positions'].items():
            for pos in positions:
                asset = pos['asset']
                if asset not in unified:
                    unified[asset] = {
                        'total': 0.0,
                        'exchange_allocation': {}
                    }
                    
                unified[asset]['total'] += float(pos['size'])
                unified[asset]['exchange_allocation'][exchange] = float(pos['size'])
                
        self.portfolio = unified
        return unified
        
    async def rebalance_portfolio(self, target_allocation: Dict[str, float]):
        """
        Rebalance portfolio across exchanges with minimal transfers
        Use CedeHub's internal settlement
        """
        current = await self.get_unified_portfolio()
        
        # Calculate current vs target
        total_value = sum(
            asset_data['total'] * await self._get_price(asset)
            for asset, asset_data in current.items()
        )
        
        rebalance_orders = []
        
        for asset, target_pct in target_allocation.items():
            target_value = total_value * target_pct
            current_value = current.get(asset, {}).get('total', 0) * await self._get_price(asset)
            
            diff = target_value - current_value
            
            if abs(diff) > total_value * 0.01:  # Rebalance if >1% off
                side = "buy" if diff > 0 else "sell"
                size = abs(diff) / await self._get_price(asset)
                
                # Find best exchange for this asset
                best_exchange = await self._find_best_exchange_for_asset(asset, side)
                
                rebalance_orders.append({
                    "exchange": best_exchange,
                    "asset": asset,
                    "side": side,
                    "size": size,
                    "type": "market",
                    "reason": "rebalancing"
                })
                
        # Execute rebalance via CedeHub
        if rebalance_orders:
            await self.cedehub._cedehub_request(
                "POST", "/v1/portfolio/rebalance",
                {"orders": rebalance_orders}
            )
            
    async def optimize_collateral(self):
        """
        Optimize collateral allocation across exchanges
        Move collateral to where it's needed most
        """
        # Get collateral utilization per exchange
        utilization = await self.cedehub._cedehub_request(
            "GET", "/v1/collateral/utilization"
        )
        
        # Find overallocation and underallocation
        transfers = []
        
        for exchange, data in utilization.items():
            util = data['utilization']
            
            if util < 0.3:  # Underutilized
                # Can move collateral out
                excess = data['available'] * 0.5  # Move 50% of excess
                transfers.append({
                    "from_exchange": exchange,
                    "asset": "USDC",
                    "amount": excess
                })
            elif util > 0.8:  # Overutilized
                # Need more collateral
                needed = data['required'] * 1.1 - data['available']
                transfers.append({
                    "to_exchange": exchange,
                    "asset": "USDC",
                    "amount": needed
                })
                
        # Execute transfers via CedeHub internal settlement
        if transfers:
            await self.cedehub._cedehub_request(
                "POST", "/v1/collateral/optimize",
                {"transfers": transfers}
            )
            
    async def tax_loss_harvesting(self):
        """
        Realize losses for tax purposes while maintaining exposure
        Sell losing positions, buy similar assets
        """
        portfolio = await self.get_unified_portfolio()
        
        harvest_orders = []
        
        for asset, data in portfolio.items():
            # Calculate average cost basis
            cost_basis = await self._get_average_cost(asset)
            current_price = await self._get_price(asset)
            
            loss_pct = (current_price - cost_basis) / cost_basis
            
            if loss_pct < -0.03:  # 3% loss
                # Sell for tax loss
                harvest_orders.append({
                    "exchange": list(data['exchange_allocation'].keys())[0],
                    "asset": asset,
                    "side": "sell",
                    "size": data['total'] * 0.5,  # Sell 50%
                    "reason": "tax_loss_harvesting"
                })
                
                # Buy correlated asset to maintain exposure
                correlated = await self._find_correlated_asset(asset)
                harvest_orders.append({
                    "exchange": "binance",  # Example
                    "asset": correlated,
                    "side": "buy",
                    "size": data['total'] * 0.5 * current_price / await self._get_price(correlated),
                    "reason": "tax_loss_harvesting_replacement"
                })
                
        # Execute via CedeHub
        if harvest_orders:
            await self.cedehub._cedehub_request(
                "POST", "/v1/tax/harvest",
                {"orders": harvest_orders}
            )
