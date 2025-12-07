"""
Strategy modules for LogicHedge
"""

from .arbitrage.triangular_arbitrage import TriangularArbitrage
from .momentum.macd_crossover import MACDStrategy, TradeSignal, Signal
from .high_frequency.market_making import MarketMakingStrategy, MarketMakingOrder

__all__ = [
    'TriangularArbitrage',
    'MACDStrategy',
    'TradeSignal',
    'Signal',
    'MarketMakingStrategy',
    'MarketMakingOrder'
]
