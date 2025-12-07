"""
Technical Analysis Strategy from Crypto-Signal
Combines multiple TA indicators with voting system
"""

import talib
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TAIndicator:
    """Technical Analysis Indicator"""
    name: str
    function: callable
    parameters: Dict
    weight: float
    buy_threshold: float
    sell_threshold: float
    
class TechnicalAnalysisStrategy:
    """
    Technical analysis with indicator voting system
    Inspired by Crypto-Signal bot
    """
    
    def __init__(self):
        self.indicators = self.initialize_indicators()
        self.voting_threshold = 0.6  # 60% agreement needed
        
    def initialize_indicators(self) -> List[TAIndicator]:
        """Initialize TA indicators"""
        indicators = []
        
        # RSI
        indicators.append(TAIndicator(
            name='RSI',
            function=self.calculate_rsi_signal,
            parameters={'period': 14, 'oversold': 30, 'overbought': 70},
            weight=1.2,
            buy_threshold=0.7,
            sell_threshold=0.3
        ))
        
        # MACD
        indicators.append(TAIndicator(
            name='MACD',
            function=self.calculate_macd_signal,
            parameters={'fast': 12, 'slow': 26, 'signal': 9},
            weight=1.0,
            buy_threshold=0.6,
            sell_threshold=0.4
        ))
        
        # Bollinger Bands
        indicators.append(TAIndicator(
            name='BBANDS',
            function=self.calculate_bbands_signal,
            parameters={'period': 20, 'std': 2.0},
            weight=0.9,
            buy_threshold=0.6,
            sell_threshold=0.4
        ))
        
        # Stochastic
        indicators.append(TAIndicator(
            name='STOCH',
            function=self.calculate_stoch_signal,
            parameters={'fastk': 14, 'slowk': 3, 'slowd': 3},
            weight=0.8,
            buy_threshold=0.7,
            sell_threshold=0.3
        ))
        
        # ADX (Trend Strength)
        indicators.append(TAIndicator(
            name='ADX',
            function=self.calculate_adx_signal,
            parameters={'period': 14},
            weight=0.7,
            buy_threshold=0.6,
            sell_threshold=0.4
        ))
        
        # Ichimoku Cloud
        indicators.append(TAIndicator(
            name='ICHIMOKU',
            function=self.calculate_ichimoku_signal,
            parameters={'tenkan': 9, 'kijun': 26, 'senkou': 52},
            weight=1.1,
            buy_threshold=0.65,
            sell_threshold=0.35
        ))
        
        return indicators
        
    def calculate_rsi_signal(self, dataframe, params: Dict) -> float:
        """Calculate RSI signal strength"""
        rsi = talib.RSI(dataframe['close'], timeperiod=params['period'])
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < params['oversold']:
            # Oversold -> buy signal
            signal_strength = 1.0 - (current_rsi / params['oversold'])
        elif current_rsi > params['overbought']:
            # Overbought -> sell signal
            signal_strength = -1.0 * ((current_rsi - params['overbought']) / (100 - params['overbought']))
        else:
            # Neutral
            signal_strength = 0
            
        return signal_strength
        
    def calculate_macd_signal(self, dataframe, params: Dict) -> float:
        """Calculate MACD signal strength"""
        macd, signal, hist = talib.MACD(
            dataframe['close'],
            fastperiod=params['fast'],
            slowperiod=params['slow'],
            signalperiod=params['signal']
        )
        
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        
        if current_macd > current_signal:
            # Bullish crossover
            signal_strength = min(1.0, (current_macd - current_signal) / abs(current_signal) if current_signal != 0 else 0.5)
        elif current_macd < current_signal:
            # Bearish crossover
            signal_strength = -min(1.0, (current_signal - current_macd) / abs(current_signal) if current_signal != 0 else 0.5)
        else:
            signal_strength = 0
            
        return signal_strength
        
    def calculate_bbands_signal(self, dataframe, params: Dict) -> float:
        """Calculate Bollinger Bands signal"""
        upper, middle, lower = talib.BBANDS(
            dataframe['close'],
            timeperiod=params['period'],
            nbdevup=params['std'],
            nbdevdn=params['std']
        )
        
        current_price = dataframe['close'].iloc[-1]
        bb_width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
        
        # Calculate %B
        percent_b = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        
        if percent_b < 0.2:
            # Near lower band -> buy signal
            signal_strength = 1.0 - (percent_b / 0.2)
        elif percent_b > 0.8:
            # Near upper band -> sell signal
            signal_strength = -1.0 * ((percent_b - 0.8) / 0.2)
        else:
            signal_strength = 0
            
        # Adjust for band width (narrow bands = stronger signal)
        signal_strength *= min(2.0, 0.02 / bb_width) if bb_width > 0 else 1.0
        
        return signal_strength
        
    async def generate_ta_signals(self, dataframe) -> Dict:
        """Generate signals from all TA indicators"""
        votes = {'buy': 0, 'sell': 0, 'hold': 0}
        weighted_votes = {'buy': 0.0, 'sell': 0.0}
        
        for indicator in self.indicators:
            try:
                signal_strength = indicator.function(dataframe, indicator.parameters)
                
                if signal_strength > indicator.buy_threshold:
                    votes['buy'] += 1
                    weighted_votes['buy'] += signal_strength * indicator.weight
                elif signal_strength < -indicator.sell_threshold:
                    votes['sell'] += 1
                    weighted_votes['sell'] += abs(signal_strength) * indicator.weight
                else:
                    votes['hold'] += 1
                    
            except Exception as e:
                print(f"Indicator {indicator.name} error: {e}")
                continue
                
        # Calculate voting results
        total_votes = sum(votes.values())
        if total_votes == 0:
            return {'signal': 'hold', 'confidence': 0, 'votes': votes}
            
        buy_ratio = votes['buy'] / total_votes
        sell_ratio = votes['sell'] / total_votes
        
        # Determine signal based on voting
        if buy_ratio >= self.voting_threshold:
            signal = 'buy'
            confidence = min(buy_ratio, weighted_votes['buy'] / len(self.indicators))
        elif sell_ratio >= self.voting_threshold:
            signal = 'sell'
            confidence = min(sell_ratio, weighted_votes['sell'] / len(self.indicators))
        else:
            signal = 'hold'
            confidence = 0
            
        return {
            'signal': signal,
            'confidence': confidence,
            'votes': votes,
            'weighted_votes': weighted_votes,
            'indicators_used': len(self.indicators)
        }
