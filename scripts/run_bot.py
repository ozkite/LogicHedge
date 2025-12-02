#!/usr/bin/env python3
"""
Main entry point for LogicHedge trading bot.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from logichedge.core.event_bus import EventBus, Event
from logichedge.strategies.market_making.hyperliquid_mm import HyperliquidMarketMaker
from logichedge.exchanges.dex.hyperliquid.client import HyperliquidClient

class LogicHedgeBot:
    """Main trading bot class"""
    
    def __init__(self, config_path: str = "configs/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.event_bus = EventBus()
        self.strategies = []
        self.exchange_clients = {}
        self.running = False
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get("file_path", "logichedge.log")),
                logging.StreamHandler()
            ]
        )
        
    async def initialize(self):
        """Initialize bot components"""
        self.logger.info("Initializing LogicHedge bot...")
        
        # Initialize exchange clients
        await self._initialize_exchanges()
        
        # Initialize strategies
        await self._initialize_strategies()
        
        # Start event bus
        await self.event_bus.start()
        
        self.logger.info("Bot initialized successfully")
        
    async def _initialize_exchanges(self):
        """Initialize exchange connections"""
        exchanges = self.config.get("exchanges", {}).get("enabled", {})
        
        # Initialize Hyperliquid if enabled
        if "hyperliquid" in exchanges.get("dex", []):
            hyperliquid_config = self.config.get("exchanges", {}).get("hyperliquid", {})
            self.exchange_clients["hyperliquid"] = HyperliquidClient(
                testnet=hyperliquid_config.get("testnet", True),
                api_key=hyperliquid_config.get("api_key"),
                private_key=hyperliquid_config.get("private_key")
            )
            self.logger.info("Hyperliquid client initialized")
            
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        strategies_config = self.config.get("strategies", {})
        
        # Initialize market making strategy if enabled
        if strategies_config.get("market_making", {}).get("enabled", False):
            mm_strategy = HyperliquidMarketMaker(
                name="hyperliquid_mm",
                event_bus=self.event_bus,
                config=strategies_config.get("market_making", {})
            )
            self.strategies.append(mm_strategy)
            self.logger.info("Market making strategy initialized")
            
    async def run(self):
        """Main bot run loop"""
        self.running = True
        self.logger.info("Starting LogicHedge bot...")
        
        # Start all strategies
        for strategy in self.strategies:
            await strategy.start()
            
        # Subscribe to system events
        self.event_bus.subscribe("system_shutdown", self.on_shutdown)
        
        # Main loop
        while self.running:
            try:
                # Placeholder for main loop logic
                # In production, this would handle reconnections, health checks, etc.
                await asyncio.sleep(1)
                
                # Log status periodically
                await self._log_status()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                
    async def _log_status(self):
        """Log system status"""
        status = {
            "running": self.running,
            "strategies": len(self.strategies),
            "exchanges": len(self.exchange_clients)
        }
        self.logger.debug(f"System status: {status}")
        
    async def on_shutdown(self, event: Event):
        """Handle shutdown event"""
        self.logger.info("Shutdown signal received")
        await self.shutdown()
        
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating shutdown...")
        self.running = False
        
        # Stop all strategies
        for strategy in self.strategies:
            await strategy.stop()
            
        # Stop event bus
        await self.event_bus.stop()
        
        self.logger.info("Shutdown complete")
        
def signal_handler(bot):
    """Handle termination signals"""
    async def handler():
        await bot.shutdown()
        sys.exit(0)
    return handler

async def main():
    """Main entry point"""
    bot = LogicHedgeBot()
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))
    
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        await bot.shutdown()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        await bot.shutdown()
        raise

if __name__ == "__main__":
    asyncio.run(main())
