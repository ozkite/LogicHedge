"""
Renko Momentum Strategy from Jesse
Profitable trend-following with Renko bricks
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RenkoConfig:
    brick_size: float = 0.001  # 0.1% brick size
    atr_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

class RenkoMomentumStrategy:
    """
    Renko-based momentum strategy
    Trades breakouts from Renko brick formations
    """
    
    def __init__(self, config: RenkoConfig = None):
        self.config = config or RenkoConfig()
        self.renko_bricks = []
        
    def calculate_renko(self, prices: List[float]) -> List[Dict]:
        """Calculate Renko bricks from price data"""
        bricks = []
        
        if not prices:
            return bricks
            
        # Start with first brick
        brick_open = prices[0]
        brick_close = brick_open
        brick_direction = 0  # 0 = neutral, 1 = up, -1 = down
        
        for price in prices[1:]:
            price_change = (price - brick_close) / brick_close
            
            if abs(price_change) >= self.config.brick_size:
                # New brick formation
                if price_change > 0:
                    # Up brick
                    new_brick = {
                        'open': brick_close,
                        'close': brick_close * (1 + self.config.brick_size),
                        'high': brick_close * (1 + self.config.brick_size),
                        'low': brick_close,
                        'direction': 1,
                        'size': self.config.brick_size
                    }
                else:
                    # Down brick
                    new_brick = {
                        'open': brick_close,
                        'close': brick_close * (1 - self.config.brick_size),
                        'high': brick_close,
                        'low': brick_close * (1 - self.config.brick_size),
                        'direction': -1,
                        'size': self.config.brick_size
                    }
                    
                bricks.append(new_brick)
                brick_close = new_brick['close']
                
        return bricks
        
    async def generate_signals(self, dataframe) -> Dict:
        """Generate trading signals based on Renko formations"""
        # Calculate Renko bricks
        prices = dataframe['close'].tolist()
        bricks = self.calculate_renko(prices)
        
        if len(bricks) < 5:
            return {'signal': 'hold'}
            
        # Analyze brick patterns
        recent_bricks = bricks[-5:]
        
        # Pattern 1: Three consecutive up bricks (strong uptrend)
        three_up = all(b['direction'] == 1 for b in recent_bricks[-3:])
        
        # Pattern 2: Three consecutive down bricks (strong downtrend)
        three_down = all(b['direction'] == -1 for b in recent_bricks[-3:])
        
        # Pattern 3: Bullish reversal (down, down, up, up, up)
        bullish_reversal = (
            recent_bricks[-5]['direction'] == -1 and
            recent_bricks[-4]['direction'] == -1 and
            recent_bricks[-3]['direction'] == 1 and
            recent_bricks[-2]['direction'] == 1 and
            recent_bricks[-1]['direction'] == 1
        )
        
        # Pattern 4: Bearish reversal (up, up, down, down, down)
        bearish_reversal = (
            recent_bricks[-5]['direction'] == 1 and
            recent_bricks[-4]['direction'] == 1 and
            recent_bricks[-3]['direction'] == -1 and
            recent_bricks[-2]['direction'] == -1 and
            recent_bricks[-1]['direction'] == -1
        )
        
        # Additional indicators
        rsi = talib.RSI(dataframe['close'], timeperiod=self.config.rsi_period).iloc[-1]
        ema_fast = talib.EMA(dataframe['close'], timeperiod=self.config.ema_fast).iloc[-1]
        ema_slow = talib.EMA(dataframe['close'], timeperiod=self.config.ema_slow).iloc[-1]
        
        # Generate signals
        if (three_up or bullish_reversal) and rsi < self.config.rsi_overbought and ema_fast > ema_slow:
            return {
                'signal': 'buy',
                'confidence': 0.8,
                'pattern': 'renko_bullish',
                'entry_price': dataframe['close'].iloc[-1],
                'stop_loss': dataframe['close'].iloc[-1] * 0.98,
                'take_profit': dataframe['close'].iloc[-1] * 1.03
            }
            
        elif (three_down or bearish_reversal) and rsi > self.config.rsi_oversold and ema_fast < ema_slow:
            return {
                'signal': 'sell',
                'confidence': 0.8,
                'pattern': 'renko_bearish',
                'entry_price': dataframe['close'].iloc[-1],
                'stop_loss': dataframe['close'].iloc[-1] * 1.02,
                'take_profit': dataframe['close'].iloc[-1] * 0.97
            }
            
        return {'signal': 'hold'}
