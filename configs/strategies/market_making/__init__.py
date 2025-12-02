"""
Market making strategy configuration module.
Configuration for various market making strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class MarketMakingType(Enum):
    """Types of market making strategies"""
    PASSIVE = "passive"          # Pure limit orders
    AGGRESSIVE = "aggressive"    # Cross spread when profitable
    ADAPTIVE = "adaptive"        # Adjust based on market conditions
    INVENTORY = "inventory"      # Focus on inventory management
    PREDICTIVE = "predictive"    # Use predictions for placement

class SpreadModel(Enum):
    """Spread calculation models"""
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    INVENTORY_ADJUSTED = "inventory_adjusted"
    COMPETITIVE = "competitive"
    ML_PREDICTIVE = "ml_predictive"

@dataclass
class SpreadConfig:
    """Spread configuration"""
    model: SpreadModel = SpreadModel.VOLATILITY_ADJUSTED
    base_spread_pct: float = 0.001  # 0.1%
    min_spread_pct: float = 0.0005  # 0.05%
    max_spread_pct: float = 0.005   # 0.5%
    
    # Volatility adjustment
    volatility_multiplier: float = 1.5
    lookback_period: int = 100  # candles
    
    # Inventory adjustment
    inventory_skew_multiplier: float = 0.0005  # 0.05% per skew unit
    
@dataclass
class OrderSizeConfig:
    """Order size configuration"""
    base_size: float = 100.0  # USD
    min_size: float = 10.0    # USD
    max_size: float = 1000.0  # USD
    
    # Dynamic sizing
    volatility_scaling: bool = True
    inventory_scaling: bool = True
    spread_scaling: bool = True
    
    # Size multipliers
    high_vol_multiplier: float = 0.5
    low_vol_multiplier: float = 1.5
    
@dataclass
class InventoryConfig:
    """Inventory management configuration"""
    target_inventory: float = 0.0
    max_inventory: float = 10000.0  # USD
    inventory_rebalance_threshold: float = 0.3  # 30% of max
    
    # Rebalancing
    rebalance_method: str = "aggressive"  # aggressive, passive, scheduled
    rebalance_interval: int = 3600  # seconds
    
    # Hedging
    hedge_enabled: bool = True
    hedge_ratio: float = 1.0
    hedge_instrument: str = "perp_futures"
    
@dataclass
class RiskConfig:
    """Risk configuration for market making"""
    max_position_usd: float = 5000.0
    max_drawdown_pct: float = 0.05  # 5%
    max_consecutive_losses: int = 5
    
    # Circuit breakers
    volatility_breaker_pct: float = 0.05  # 5% move in 5 minutes
    volume_breaker_multiplier: float = 10.0  # 10x average volume
    loss_breaker_usd: float = 1000.0
    
@dataclass
class MarketMakingConfig:
    """Complete market making configuration"""
    # General settings
    strategy_type: MarketMakingType
    enabled: bool = True
    name: str = "market_maker_01"
    symbols: List[str] = None
    
    # Spread configuration
    spread_config: SpreadConfig = None
    
    # Order size configuration
    order_size_config: OrderSizeConfig = None
    
    # Inventory management
    inventory_config: InventoryConfig = None
    
    # Risk management
    risk_config: RiskConfig = None
    
    # Execution settings
    update_frequency: int = 1  # seconds
    cancel_threshold_pct: float = 0.001  # Cancel if price moves 0.1%
    replace_threshold_pct: float = 0.0005  # Replace if price moves 0.05%
    
    # Advanced features
    use_predictive_placement: bool = False
    adverse_selection_protection: bool = True
    anti_gaming_measures: bool = True
    
    # Monitoring
    log_fills: bool = True
    log_quotes: bool = False
    performance_report_interval: int = 300  # seconds
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-USDT", "ETH-USDT"]
        if self.spread_config is None:
            self.spread_config = SpreadConfig()
        if self.order_size_config is None:
            self.order_size_config = OrderSizeConfig()
        if self.inventory_config is None:
            self.inventory_config = InventoryConfig()
        if self.risk_config is None:
            self.risk_config = RiskConfig()
