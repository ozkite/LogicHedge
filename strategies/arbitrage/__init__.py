"""
Arbitrage strategy configuration module.
Configuration for various arbitrage strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class ArbitrageType(Enum):
    """Types of arbitrage strategies"""
    STATISTICAL = "statistical"
    TRIANGULAR = "triangular"
    CROSS_EXCHANGE = "cross_exchange"
    FUNDING_RATE = "funding_rate"
    FUTURES_BASIS = "futures_basis"
    DEX_CEX = "dex_cex"
    FLASH_LOAN = "flash_loan"

class ExecutionMode(Enum):
    """Execution modes for arbitrage"""
    ATOMIC = "atomic"          # All trades execute simultaneously
    SEQUENTIAL = "sequential"  # Trades execute in sequence
    HEDGED = "hedged"          # Hedge with derivatives

@dataclass
class TriangularArbitrageConfig:
    """Configuration for triangular arbitrage"""
    enabled: bool = True
    pairs: List[Tuple[str, str, str]] = None  # (A/B, B/C, C/A)
    min_profit_pct: float = 0.001  # 0.1%
    max_slippage_pct: float = 0.002  # 0.2%
    max_position_size: float = 1000.0
    
    def __post_init__(self):
        if self.pairs is None:
            self.pairs = [
                ("BTC/USDT", "ETH/BTC", "ETH/USDT"),
                ("SOL/USDT", "BTC/SOL", "BTC/USDT")
            ]

@dataclass
class CrossExchangeConfig:
    """Configuration for cross-exchange arbitrage"""
    enabled: bool = True
    exchange_pairs: List[Tuple[str, str]] = None  # (exchange1, exchange2)
    symbols: List[str] = None
    min_price_diff_pct: float = 0.002  # 0.2%
    max_transfer_time: int = 30  # seconds
    use_stablecoin_bridge: bool = True
    
    def __post_init__(self):
        if self.exchange_pairs is None:
            self.exchange_pairs = [("binance", "bybit"), ("okx", "kucoin")]
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

@dataclass
class FundingRateArbitrageConfig:
    """Configuration for funding rate arbitrage"""
    enabled: bool = True
    symbols: List[str] = None
    min_funding_rate: float = 0.0005  # 0.05%
    hedge_ratio: float = 1.0
    max_leverage: float = 3.0
    rebalance_threshold: float = 0.2
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-USDT-PERP", "ETH-USDT-PERP"]

@dataclass
class ArbitrageConfig:
    """Complete arbitrage strategy configuration"""
    # General settings
    strategy_type: ArbitrageType
    enabled: bool = True
    name: str = "arbitrage_01"
    
    # Profit thresholds
    min_profit_pct: float = 0.001
    min_profit_usd: float = 5.0
    max_slippage_pct: float = 0.002
    
    # Position sizing
    position_sizing: str = "fixed"  # fixed, percentage, kelly
    fixed_size: float = 500.0
    max_position_usd: float = 5000.0
    
    # Execution
    execution_mode: ExecutionMode = ExecutionMode.ATOMIC
    max_execution_time: int = 5  # seconds
    use_flash_loans: bool = False
    gas_limit_multiplier: float = 1.5
    
    # Risk management
    max_daily_trades: int = 20
    max_concurrent_opportunities: int = 3
    stop_loss_pct: float = 0.005
    correlation_threshold: float = 0.8
    
    # Specific strategy configs
    triangular_config: Optional[TriangularArbitrageConfig] = None
    cross_exchange_config: Optional[CrossExchangeConfig] = None
    funding_rate_config: Optional[FundingRateArbitrageConfig] = None
    
    # Monitoring
    log_all_opportunities: bool = True
    alert_on_large_opportunity: bool = True
    opportunity_threshold_pct: float = 0.01
