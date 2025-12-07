#!/usr/bin/env python3
"""
Main CCXT trading bot for Logic Hedge
"""

import asyncio
import logging
import yaml
from pathlib import Path
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ccxt_bot.log'),
        logging.StreamHandler()
    ]
