"""
Alpha Streams Strategy from QuantConnect Lean
Combines multiple alpha models for higher accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class AlphaType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"

@dataclass
class AlphaModel:
    """Individual alpha model"""
    name: str
    alpha_type: AlphaType
    weight: float
    lookback_period: int
    rebalance_frequency: str  # 'daily', 'hourly', 'minute'
    
class AlphaStreamsStrategy:
    """
    Combines multiple alpha models into a stream
    Inspired by QuantConnect's Alpha Streams
    """
    
    def __init__(self):
        self.alpha_models = []
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize alpha models"""
        # Momentum Alpha
        self.alpha_models.append(AlphaModel(
            name="MomentumAlpha",
            alpha_type=AlphaType.MOMENTUM,
            weight=0.30,
            lookback_period=20,
            rebalance_frequency='daily'
        ))
        
        # Mean Reversion Alpha
        self.alpha_models.append(AlphaModel(
            name="MeanReversionAlpha",
            alpha_type=AlphaType.MEAN_REVERSION,
            weight=0.25,
            lookback_period=10,
            rebalance_frequency='daily'
        ))
        
        # Volatility Alpha
        self.alpha_models.append(AlphaModel(
            name="VolatilityAlpha",
            alpha_type=AlphaType.VOLATILITY,
            weight=0.20,
            lookback_period=30,
            rebalance_frequency='daily'
        ))
        
        # Quality Alpha
        self.alpha_models.append(AlphaModel(
            name="QualityAlpha",
            alpha_type=AlphaType.QUALITY,
            weight=0.15,
            lookback_period=60,
            rebalance_frequency='daily'
        ))
        
        # Value Alpha
        self.alpha_models.append(AlphaModel(
            name="ValueAlpha",
            alpha_type=AlphaType.VALUE,
            weight=0.10,
            lookback_period=90,
            rebalance_frequency='daily'
        ))
        
    async def calculate_momentum_alpha(self, dataframe, lookback: int) -> float:
        """Calculate momentum alpha score"""
        returns = dataframe['close'].pct_change(lookback)
        volatility = returns.rolling(lookback).std()
        
        # Sharpe-like momentum score
        if volatility.iloc[-1] > 0:
            momentum_score = returns.iloc[-1] / volatility.iloc[-1]
        else:
            momentum_score = 0
            
        return momentum_score
        
    async def calculate_mean_reversion_alpha(self, dataframe, lookback: int) -> float:
        """Calculate mean reversion alpha score"""
        # Z-score of price from moving average
        ma = dataframe['close'].rolling(lookback).mean()
        std = dataframe['close'].rolling(lookback).std()
        
        if std.iloc[-1] > 0:
            z_score = (dataframe['close'].iloc[-1] - ma.iloc[-1]) / std.iloc[-1]
        else:
            z_score = 0
            
        # Mean reversion expects price to revert to mean
        # Negative z-score = buy signal, positive = sell signal
        return -z_score  # Negative for mean reversion
        
    async def calculate_volatility_alpha(self, dataframe, lookback: int) -> float:
        """Calculate volatility alpha score"""
        returns = dataframe['close'].pct_change()
        volatility = returns.rolling(lookback).std()
        
        # Volatility forecasting (GARCH-like)
        recent_vol = volatility.iloc[-lookback:].values
        if len(recent_vol) > 1:
            vol_trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
        else:
            vol_trend = 0
            
        # Negative alpha for increasing volatility (risk-off)
        return -vol_trend
        
    async def generate_alpha_stream(self, dataframe) -> Dict:
        """Generate combined alpha stream"""
        alpha_scores = {}
        total_score = 0
        
        for model in self.alpha_models:
            if model.alpha_type == AlphaType.MOMENTUM:
                score = await self.calculate_momentum_alpha(dataframe, model.lookback_period)
            elif model.alpha_type == AlphaType.MEAN_REVERSION:
                score = await self.calculate_mean_reversion_alpha(dataframe, model.lookback_period)
            elif model.alpha_type == AlphaType.VOLATILITY:
                score = await self.calculate_volatility_alpha(dataframe, model.lookback_period)
            elif model.alpha_type == AlphaType.QUALITY:
                # Quality based on volume consistency
                volume_std = dataframe['volume'].rolling(model.lookback_period).std()
                volume_mean = dataframe['volume'].rolling(model.lookback_period).mean()
                score = 1 / (volume_std.iloc[-1] / volume_mean.iloc[-1]) if volume_mean.iloc[-1] > 0 else 0
            elif model.alpha_type == AlphaType.VALUE:
                # Simple value metric (price relative to moving average)
                ma = dataframe['close'].rolling(model.lookback_period).mean()
                score = (ma.iloc[-1] - dataframe['close'].iloc[-1]) / dataframe['close'].iloc[-1]
            else:
                score = 0
                
            weighted_score = score * model.weight
            alpha_scores[model.name] = weighted_score
            total_score += weighted_score
            
        # Normalize total score to -1 to 1 range
        normalized_score = np.tanh(total_score)
        
        # Generate signal
        if normalized_score > 0.2:
            signal = 'buy'
            confidence = min(abs(normalized_score), 0.9)
        elif normalized_score < -0.2:
            signal = 'sell'
            confidence = min(abs(normalized_score), 0.9)
        else:
            signal = 'hold'
            confidence = 0
            
        return {
            'signal': signal,
            'confidence': confidence,
            'total_alpha': normalized_score,
            'alpha_breakdown': alpha_scores,
            'timestamp': pd.Timestamp.now()
        }
