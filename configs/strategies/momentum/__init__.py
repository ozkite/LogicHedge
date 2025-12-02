"""
Momentum strategy configuration module.
Configuration for momentum and trend-following strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class MomentumType(Enum):
    """Types of momentum strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM_CROSSOVER = "momentum_crossover"
    VOLATILITY_MOMENTUM = "volatility_momentum"

class SignalType(Enum):
    """Types of trading signals"""
    SINGLE_INDICATOR = "single_indicator"
    MULTI_INDICATOR = "multi_indicator"
    ML_PREDICTION = "ml_prediction"
    ENSEMBLE = "ensemble"

@dataclass
class IndicatorConfig:
    """Technical indicator configuration"""
    name: str
    parameters: Dict[str, float]
    weight: float = 1.0
    enabled: bool = True
    
@dataclass
class TrendDetectionConfig:
    """Trend detection configuration"""
    enabled: bool = True
    methods: List[str] = None  # moving_average, adx, parabolic_sar
    confirmation_periods: int = 3
    min_trend_strength: float = 25.0  # ADX value
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["moving_average", "adx", "macd"]

@dataclass
class EntryConfig:
    """Entry signal configuration"""
    signal_type: SignalType = SignalType.MULTI_INDICATOR
    indicators: List
