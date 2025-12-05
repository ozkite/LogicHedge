#!/usr/bin/env python3
"""
Start the safe arbitrage bot with your exchanges
"""

import asyncio
import yaml
from logichedge.strategies.safe_arbitrage_bot import SafeArbitrageBot

async def main():
    # Load config
    with open('configs/exchanges/api_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize bot
    bot = SafeArbitrageBot(config)
    await bot.initialize()
    
    # Start strategies
    tasks = [
        asyncio.create_task(bot.run_cross_exchange_arbitrage()),
        asyncio.create_task(bot.run_funding_rate_arbitrage()),
    ]
    
    # Run performance monitor
    async def monitor():
        while True:
            perf = bot.get_performance()
            print(f"Daily P&L: ${perf['daily_pnl']:.2f}, Trades: {perf['total_trades']}")
            await asyncio.sleep(60)
    
    tasks.append(asyncio.create_task(monitor()))
    
    # Run forever
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
