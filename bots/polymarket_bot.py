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
            gas_limit=polymarket_config['
