#!/usr/bin/env python3
"""
Start CedeHub-powered multi-exchange trading bot
"""

import asyncio
import yaml
from logichedge.integrations.cedehub.strategies import CedeHubArbitrage
from logichedge.strategies.cedehub_market_making import CedeHubMarketMaker
from logichedge.integrations.cedehub.portfolio_manager import CedeHubPortfolioManager

async def main():
    # Load CedeHub config
    with open('configs/cedehub.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize CedeHub client
    cedehub = CedeHubArbitrage(
        api_key=config['api_key'],
        secret=config['api_secret']
    )
    
    await cedehub.initialize()
    
    # Start strategies
    strategies = []
    
    # 1. Unified arbitrage
    strategies.append(
        asyncio.create_task(cedehub.execute_unified_arbitrage())
    )
    
    # 2. Cross-collateral trading
    strategies.append(
        asyncio.create_task(cedehub.execute_cross_collateral_strategy())
    )
    
    # 3. Portfolio hedging
    strategies.append(
        asyncio.create_task(cedehub.execute_portfolio_hedging())
    )
    
    # 4. Market making
    mm = CedeHubMarketMaker(cedehub, ['BTC/USDT', 'ETH/USDT'])
    strategies.append(
        asyncio.create_task(mm.run_unified_market_making())
    )
    
    # 5. Portfolio management
    pm = CedeHubPortfolioManager(cedehub)
    strategies.append(
        asyncio.create_task(pm.optimize_collateral())
    )
    
    print("🎯 CedeHub Bot Started")
    print(f"💰 Unified Balance: ${cedehub.unified_balance:,.2f}")
    print(f"📊 Connected Exchanges: {cedehub.connected_exchanges}")
    
    # Run forever
    await asyncio.gather(*strategies)

if __name__ == '__main__':
    asyncio.run(main())
