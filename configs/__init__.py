"""
Configuration module for LogicHedge.
Centralizes configuration loading and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    enabled: bool = False
    rate_limit: int = 10
    
@dataclass  
class StrategyConfig:
    """Strategy-specific configuration"""
    name: str
    enabled: bool = False
    max_position_size: float = 1000.0
    min_profit_threshold: float = 0.001

class ConfigManager:
    """Manages loading and accessing configuration files"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._config_cache = {}
        
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.config_dir / filename
        
        if filename in self._config_cache:
            return self._config_cache[filename]
            
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                self._config_cache[filename] = config
                logger.info(f"Loaded config: {filename}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {filename}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config {filename}: {e}")
            raise
            
    def get_main_config(self) -> Dict[str, Any]:
        """Get main configuration"""
        return self.load_config("main_config.yaml")
        
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk configuration"""
        return self.load_config("risk_config.yaml")
        
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy-specific configuration"""
        strategy_file = f"strategies/{strategy_name}.yaml"
        return self.load_config(strategy_file)
        
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get exchange-specific configuration"""
        exchange_file = f"exchanges/{exchange_name}.yaml"
        return self.load_config(exchange_file)
        
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all strategies"""
        strategies = {}
        strategy_dir = self.config_dir / "strategies"
        
        if strategy_dir.exists():
            for file in strategy_dir.glob("*.yaml"):
                strategy_name = file.stem
                strategies[strategy_name] = self.load_config(f"strategies/{file.name}")
                
        return strategies

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration manager"""
    return config_manager
