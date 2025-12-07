"""
Pairs Trading Strategy
Statistical arbitrage between correlated assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import coint

class PairsTradingStrategy:
    """
    Pairs trading - market neutral strategy
    Trades the spread between two cointegrated assets
    """
    
    def __init__(self, lookback_period: int = 60, zscore_threshold: float = 2.0):
        self.lookback = lookback_period
        self.zscore_threshold = zscore_threshold
        self.cointegrated_pairs = []
        self.spread_history = {}
        
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, 
                               significance_level: float = 0.05) -> List[Tuple[str, str]]:
        """Find cointegrated pairs using Johansen test"""
        symbols = price_data.columns
        pairs = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                
                # Get price series
                series1 = price_data[symbol1].dropna()
                series2 = price_data[symbol2].dropna()
                
                # Align series
                common_index = series1.index.intersection(series2.index)
                if len(common_index) < self.lookback:
                    continue
                    
                series1 = series1.loc[common_index]
                series2 = series2.loc[common_index]
                
                # Test for cointegration
                score, pvalue, _ = coint(series1, series2)
                
                if pvalue < significance_level:
                    # Calculate hedge ratio (OLS regression)
                    hedge_ratio = np.polyfit(series1, series2, 1)[0]
                    
                    pairs.append({
                        'pair': (symbol1, symbol2),
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'score': score
                    })
                    
        # Sort by p-value (most significant first)
        pairs.sort(key=lambda x: x['pvalue'])
        self.cointegrated_pairs = pairs[:10]  # Top 10 pairs
        
        return self.cointegrated_pairs
        
    def calculate_spread(self, price1: pd.Series, price2: pd.Series, 
                        hedge_ratio: float) -> pd.Series:
        """Calculate spread between two price series"""
        spread = price1 - (hedge_ratio * price2)
        return spread
        
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of spread"""
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        zscore = (spread - mean) / std
        return zscore
        
    async def generate_signals(self, price_data: pd.DataFrame) -> Dict:
        """Generate pairs trading signals"""
        if not self.cointegrated_pairs:
            self.find_cointegrated_pairs(price_data)
            
        signals = {}
        
        for pair_info in self.cointegrated_pairs:
            symbol1, symbol2 = pair_info['pair']
            hedge_ratio = pair_info['hedge_ratio']
            
            # Get current prices
            price1 = price_data[symbol1].iloc[-self.lookback:]
            price2 = price_data[symbol2].iloc[-self.lookback:]
            
            if len(price1) < self.lookback or len(price2) < self.lookback:
                continue
                
            # Calculate spread and z-score
            spread = self.calculate_spread(price1, price2, hedge_ratio)
            zscore = self.calculate_zscore(spread)
            current_zscore = zscore.iloc[-1]
            
            # Store spread history
            pair_key = f"{symbol1}_{symbol2}"
            self.spread_history[pair_key] = spread
            
            # Generate signals
            if current_zscore > self.zscore_threshold:
                # Spread is wide - sell spread (sell asset1, buy asset2)
                signals[pair_key] = {
                    'signal': 'sell_spread',
                    'zscore': current_zscore,
                    'action1': ('sell', symbol1),
                    'action2': ('buy', symbol2),
                    'hedge_ratio': hedge_ratio,
                    'entry_zscore': current_zscore,
                    'target_zscore': 0,  # Mean reversion target
                    'stop_zscore': current_zscore * 1.5  # Stop if spreads widen further
                }
                
            elif current_zscore < -self.zscore_threshold:
                # Spread is narrow - buy spread (buy asset1, sell asset2)
                signals[pair_key] = {
                    'signal': 'buy_spread',
                    'zscore': current_zscore,
                    'action1': ('buy', symbol1),
                    'action2': ('sell', symbol2),
                    'hedge_ratio': hedge_ratio,
                    'entry_zscore': current_zscore,
                    'target_zscore': 0,
                    'stop_zscore': current_zscore * 1.5
                }
                
        return signals
