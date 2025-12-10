"""
Main Polymarket trading bot
"""

import asyncio
import yaml
import logging
from typing import Dict, List
from datetime import datetime
from strategies.polymarket.prediction_market import (
    PolymarketTradingBot, 
    TradingSignal
)
from utils.polymarket_utils import PolymarketAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolymarketTradingLoop:
    """Main trading loop for Polymarket strategies"""
    
    def __init__(self, config_path: str = "config/polymarket_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize bot
        polymarket_config = self.config['polymarket']
        
        self.bot = PolymarketTradingBot(
            web3_provider=polymarket_config['web3_provider'],
            private_key=None,  # Set in production
            polymarket_address=polymarket_config['contracts']['polymarket'],
            conditional_tokens_address=polymarket_config['contracts']['conditional_tokens'],
            gas_limit=polymarket_config['trading']['gas_limit']
        )
        
        # Initialize API
        self.api = PolymarketAPI()
        
        # State
        self.active_strategies = []
        self.running = False
        
    async def run_strategies(self):
        """Run all active strategies"""
        strategies_config = self.config['polymarket']['strategies']
        
        tasks = []
        
        if strategies_config['arbitrage']['enabled']:
            tasks.append(self.run_arbitrage_strategy())
        
        if strategies_config['spike_detection']['enabled']:
            tasks.append(self.run_spike_detection_strategy())
        
        if strategies_config['copy_trading']['enabled']:
            tasks.append(self.run_copy_trading_strategy())
        
        if strategies_config['market_making']['enabled']:
            tasks.append(self.run_market_making_strategy())
        
        # Run all strategies concurrently
        if tasks:
            await asyncio.gather(*tasks)
    
    async def run_arbitrage_strategy(self):
        """Run arbitrage strategy"""
        logger.info("Running arbitrage strategy")
        
        try:
            signals = await self.bot.arbitrage_opportunities()
            
            for signal in signals:
                if signal.expected_profit > self.config['polymarket']['strategies']['arbitrage']['min_profit']:
                    logger.info(f"Arbitrage opportunity found: {signal}")
                    
                    # Execute trade
                    result = await self.bot.execute_trade(signal)
                    if result['status'] == 'success':
                        logger.info(f"Arbitrage trade executed: {result}")
        
        except Exception as e:
            logger.error(f"Arbitrage strategy error: {e}")
    
    async def run_spike_detection_strategy(self):
        """Run spike detection strategy"""
        logger.info("Running spike detection strategy")
        
        try:
            signals = await self.bot.spike_detection_strategy()
            
            for signal in signals:
                if signal.confidence > 0.6:  # Minimum confidence threshold
                    logger.info(f"Spike detected: {signal}")
                    
                    result = await self.bot.execute_trade(signal)
                    if result['status'] == 'success':
                        logger.info(f"Spike trade executed: {result}")
        
        except Exception as e:
            logger.error(f"Spike detection strategy error: {e}")
    
    async def run_copy_trading_strategy(self):
        """Run copy trading strategy"""
        logger.info("Running copy trading strategy")
        
        try:
            trader_addresses = self.config['polymarket']['copy_traders']
            signals = await self.bot.copy_trading_strategy(trader_addresses)
            
            for signal in signals:
                logger.info(f"Copy trade signal: {signal}")
                
                result = await self.bot.execute_trade(signal)
                if result['status'] == 'success':
                    logger.info(f"Copy trade executed: {result}")
        
        except Exception as e:
            logger.error(f"Copy trading strategy error: {e}")
    
    async def run_market_making_strategy(self):
        """Run market making strategy"""
        logger.info("Running market making strategy")
        
        try:
            # Get active markets
            async with self.api as api:
                markets = await api.fetch_markets(first=10)
            
            for market in markets:
                signals = await self.bot.market_making_strategy(
                    market['id'],
                    spread=self.config['polymarket']['strategies']['market_making']['spread']
                )
                
                for signal in signals:
                    logger.info(f"Market making signal: {signal}")
                    
                    result = await self.bot.execute_trade(signal)
                    if result['status'] == 'success':
                        logger.info(f"Market making trade executed: {result}")
        
        except Exception as e:
            logger.error(f"Market making strategy error: {e}")
    
    async def monitor_and_manage(self):
        """Monitor positions and manage risk"""
        while self.running:
            try:
                # Monitor positions
                positions_status = await self.bot.monitor_positions()
                
                for position in positions_status:
                    if position['should_close']:
                        logger.info(f"Closing position: {position['close_reason']}")
                        # Implement position closing logic
                
                # Risk management checks
                if not self.risk_management_check():
                    logger.warning("Risk limits breached - pausing trading")
                    self.running = False
                    break
                
                # Wait before next check
                await asyncio.sleep(
                    self.config['polymarket']['monitoring']['update_interval']
                )
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    def risk_management_check(self) -> bool:
        """Check risk management limits"""
        # Implement risk checks based on config
        return True
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting Polymarket trading bot")
        self.running = True
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_and_manage())
        
        try:
            while self.running:
                logger.info("Running strategy cycle")
                
                # Run all strategies
                await self.run_strategies()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Run every 30 seconds
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.running = False
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            logger.info("Polymarket trading bot stopped")

async def main():
    """Main entry point"""
    bot = PolymarketTradingLoop()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
