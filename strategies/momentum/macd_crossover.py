"""
MACD Crossover Strategy
Mid Risk - Momentum based strategy using MACD indicators
Source: Inspired by QuantConnect/Lean and jesse repos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradeSignal:
    signal: Signal
    strength: float
    price: float
    timestamp: pd.Timestamp
    macd: float
    signal_line: float
    histogram: float

class MACDStrategy:
    def __init__(self, 
                 fast_period=12, 
                 slow_period=26, 
                 signal_period=9,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30):
        """
        MACD Crossover Strategy with RSI confirmation
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            rsi_period: RSI period for confirmation
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD indicator"""
        df = pd.DataFrame({'close': prices})
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # Calculate Signal line
        df['signal'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate Histogram
        df['histogram'] = df['macd'] - df['signal']
        
        return df
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, 
                        prices: pd.Series, 
                        volume: Optional[pd.Series] = None) -> List[TradeSignal]:
        """
        Generate trading signals based on MACD crossover with RSI confirmation
        """
        signals = []
        
        # Calculate indicators
        macd_df = self.calculate_macd(prices)
        rsi = self.calculate_rsi(prices)
        
        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            current_macd = macd_df['macd'].iloc[i]
            current_signal = macd_df['signal'].iloc[i]
            current_hist = macd_df['histogram'].iloc[i]
            prev_hist = macd_df['histogram'].iloc[i-1]
            current_rsi = rsi.iloc[i] if i < len(rsi) else 50
            
            signal_strength = 0.0
            trade_signal = Signal.HOLD
            
            # MACD Crossover signals
            # Bullish crossover: MACD crosses above Signal line
            if (current_macd > current_signal and 
                macd_df['macd'].iloc[i-1] <= macd_df['signal'].iloc[i-1]):
                
                # RSI confirmation (not overbought)
                if current_rsi < self.rsi_overbought:
                    # Strength based on histogram momentum
                    signal_strength = abs(current_hist) / current_price
                    if current_hist > 0 and prev_hist <= 0:  # Histogram turning positive
                        signal_strength *= 1.5
                    
                    trade_signal = Signal.BUY
            
            # Bearish crossover: MACD crosses below Signal line
            elif (current_macd < current_signal and 
                  macd_df['macd'].iloc[i-1] >= macd_df['signal'].iloc[i-1]):
                
                # RSI confirmation (not oversold)
                if current_rsi > self.rsi_oversold:
                    signal_strength = abs(current_hist) / current_price
                    if current_hist < 0 and prev_hist >= 0:  # Histogram turning negative
                        signal_strength *= 1.5
                    
                    trade_signal = Signal.SELL
            
            # Divergence detection (additional signal strength)
            if i > 10:
                # Price making higher highs but MACD making lower highs (bearish divergence)
                price_higher = prices.iloc[i] > prices.iloc[i-10:i].max()
                macd_lower = current_macd < macd_df['macd'].iloc[i-10:i].max()
                
                if price_higher and macd_lower and trade_signal == Signal.SELL:
                    signal_strength *= 2.0
            
            if trade_signal != Signal.HOLD:
                signal = TradeSignal(
                    signal=trade_signal,
                    strength=min(signal_strength, 1.0),  # Cap at 1.0
                    price=current_price,
                    timestamp=prices.index[i],
                    macd=current_macd,
                    signal_line=current_signal,
                    histogram=current_hist
                )
                signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, 
                               capital: float, 
                               signal: TradeSignal, 
                               risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on signal strength and risk management
        
        Args:
            capital: Available capital
            signal: Trade signal
            risk_per_trade: Max risk per trade (2% default)
        
        Returns:
            Position size in base currency
        """
        base_position = capital * risk_per_trade
        
        # Adjust position size based on signal strength
        adjusted_position = base_position * (0.5 + signal.strength * 0.5)
        
        # Cap at 10% of capital for single trade
        max_position = capital * 0.1
        
        return min(adjusted_position, max_position)
