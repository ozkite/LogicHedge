"""
AI Agents strategy configuration module.
Contains configurations for various AI-driven trading strategies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class AgentType(Enum):
    """Types of AI agents"""
    REINFORCEMENT_LEARNING = "rl"
    DEEP_LEARNING = "dl"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    SWARM = "swarm"

class ActionSpace(Enum):
    """Action spaces for RL agents"""
    DISCRETE = "discrete"      # Buy/Hold/Sell
    CONTINUOUS = "continuous"  # Percentage allocation
    MULTI_DISCRETE = "multi_discrete"  # Multiple assets

@dataclass
class ModelConfig:
    """AI model configuration"""
    model_type: AgentType
    model_path: str
    input_features: List[str]
    output_dim: int
    hidden_layers: List[int] = None
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    batch_size: int = 32
    sequence_length: int = 60  # For time series models
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]

@dataclass
class TrainingConfig:
    """Training configuration for AI agents"""
    episodes: int = 1000
    steps_per_episode: int = 1000
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_frequency: int = 100
    replay_buffer_size: int = 10000
    warmup_steps: int = 1000
    
@dataclass
class SwarmConfig:
    """Swarm intelligence configuration"""
    num_agents: int = 10
    communication_topology: str = "fully_connected"  # star, ring, fully_connected
    consensus_threshold: float = 0.7
    diversity_penalty: float = 0.1
    exploration_rate: float = 0.1
    
@dataclass
class AIConfig:
    """Complete AI agent configuration"""
    # General settings
    enabled: bool = True
    name: str = "ai_agent_01"
    symbols: List[str] = None
    
    # Model settings
    model_config: ModelConfig = None
    training_config: TrainingConfig = None
    
    # Swarm settings (if applicable)
    swarm_config: Optional[SwarmConfig] = None
    
    # Execution settings
    max_position_size: float = 1000.0
    min_confidence: float = 0.65
    action_delay_seconds: int = 1
    
    # Risk settings
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_daily_trades: int = 50
    
    # Monitoring
    log_predictions: bool = True
    save_model_checkpoints: bool = True
    checkpoint_interval: int = 100
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC-USDT", "ETH-USDT"]
        if self.model_config is None:
            self.model_config = ModelConfig(
                model_type=AgentType.REINFORCEMENT_LEARNING,
                model_path="models/rl_agent_v1",
                input_features=["price", "volume", "rsi", "macd", "bb_width"],
                output_dim=3  # Buy/Hold/Sell
            )
        if self.training_config is None:
            self.training_config = TrainingConfig()
