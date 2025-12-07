"""
CombinedBinHAndCluc strategy from Freqtrade
Combines multiple indicator systems for high confidence signals
"""

class CombinedBinHAndClucStrategy:
    """
    Combined strategy using:
    - Bollinger Bands
    - Heikin-Ashi
    - CLUC (Close, Low, Upper Close) patterns
    """
    
    def __init__(self):
        self.name = "CombinedBinHAndCluc"
        self.config = {
            # Bollinger Bands parameters
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_gain': 0.02,  # 2% from lower band
            
            # Heikin-Ashi parameters
            'ha_trend_period': 3,
            
            # CLUC parameters
            'cluc_period': 12,
            'cluc_std': 1.0,
            
            # Volume
            'volume_threshold': 1.2,
            
            # RSI
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_buy': 40,
            
            # Timeframes
            'main_tf': '5m',
            'confirm_tf': '15m'
        }
        
    async def calculate_indicators(self, dataframe):
        """Calculate all indicators"""
        # Bollinger Bands
        dataframe['bb_lower'], dataframe['bb_middle'], dataframe['bb_upper'] = \
            talib.BBANDS(dataframe['close'], 
                        timeperiod=self.config['bb_period'],
                        nbdevup=self.config['bb_std'],
                        nbdevdn=self.config['bb_std'])
        
        # Heikin-Ashi
        dataframe = self.calculate_heikin_ashi(dataframe)
        
        # CLUC (Close, Low, Upper Close) indicator
        dataframe = self.calculate_cluc(dataframe)
        
        # RSI
        dataframe['rsi'] = talib.RSI(dataframe['close'], 
                                    timeperiod=self.config['rsi_period'])
        
        # Volume
        dataframe['volume_sma'] = talib.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        return dataframe
        
    def calculate_heikin_ashi(self, dataframe):
        """Calculate Heikin-Ashi candles"""
        ha_close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        # Heikin-Ashi open
        ha_open = (dataframe['open'].shift(1) + ha_close.shift(1)) / 2
        ha_open.iloc[0] = (dataframe['open'].iloc[0] + dataframe['close'].iloc[0]) / 2
        
        # Heikin-Ashi high/low
        ha_high = dataframe[['high', 'open', 'close']].max(axis=1)
        ha_low = dataframe[['low', 'open', 'close']].min(axis=1)
        
        dataframe['ha_open'] = ha_open
        dataframe['ha_close'] = ha_close
        dataframe['ha_high'] = ha_high
        dataframe['ha_low'] = ha_low
        
        # Heikin-Ashi trend
        dataframe['ha_trend'] = (
            (ha_close > ha_open) &
            (ha_close.shift(1) > ha_open.shift(1)) &
            (ha_close.shift(2) > ha_open.shift(2))
        ).astype(int)
        
        return dataframe
        
    def calculate_cluc(self, dataframe):
        """Calculate CLUC indicator"""
        # Close > Upper SMA
        upper_sma = talib.SMA(dataframe['close'], timeperiod=self.config['cluc_period'])
        dataframe['close_gt_upper'] = dataframe['close'] > upper_sma * (1 + self.config['cluc_std']/100)
        
        # Close > Lower SMA
        lower_sma = talib.SMA(dataframe['low'], timeperiod=self.config['cluc_period'])
        dataframe['close_gt_lower'] = dataframe['close'] > lower_sma
        
        # CLUC signal
        dataframe['cluc_signal'] = dataframe['close_gt_upper'] & dataframe['close_gt_lower']
        
        return dataframe
        
    async def generate_buy_signals(self, dataframe):
        """Generate buy signals with multiple confirmations"""
        conditions = []
        
        # Condition 1: Price near Bollinger Band lower
        bb_condition = (
            (dataframe['close'] <= dataframe['bb_lower'] * (1 + self.config['bb_gain'])) &
            (dataframe['close'].shift(1) > dataframe['bb_lower'].shift(1) * (1 + self.config['bb_gain']))
        )
        conditions.append(bb_condition)
        
        # Condition 2: Heikin-Ashi bullish trend
        ha_condition = dataframe['ha_trend'] == 1
        conditions.append(ha_condition)
        
        # Condition 3: CLUC signal
        cluc_condition = dataframe['cluc_signal']
        conditions.append(cluc_condition)
        
        # Condition 4: RSI oversold or recovering
        rsi_condition = (
            (dataframe['rsi'] < self.config['rsi_oversold']) |
            ((dataframe['rsi'] > self.config['rsi_buy']) & (dataframe['rsi'].shift(1) <= self.config['rsi_buy']))
        )
        conditions.append(rsi_condition)
        
        # Condition 5: Volume confirmation
        volume_condition = dataframe['volume_ratio'] > self.config['volume_threshold']
        conditions.append(volume_condition)
        
        # Combine all conditions
        dataframe['buy_signal'] = np.all(conditions, axis=0)
        
        # Calculate buy price (slightly above current for limit orders)
        dataframe['buy_price'] = dataframe['close'] * 1.001  # 0.1% above
        
        return dataframe
        
    async def generate_sell_signals(self, dataframe):
        """Generate sell signals"""
        conditions = []
        
        # Condition 1: Price near Bollinger Band upper
        bb_sell_condition = (
            (dataframe['close'] >= dataframe['bb_upper'] * 0.99) &
            (dataframe['close'].shift(1) < dataframe['bb_upper'].shift(1) * 0.99)
        )
        conditions.append(bb_sell_condition)
        
        # Condition 2: Heikin-Ashi bearish
        ha_sell_condition = (
            (dataframe['ha_close'] < dataframe['ha_open']) &
            (dataframe['ha_close'].shift(1) < dataframe['ha_open'].shift(1))
        )
        conditions.append(ha_sell_condition)
        
        # Condition 3: RSI overbought
        rsi_sell_condition = dataframe['rsi'] > 80
        conditions.append(rsi_sell_condition)
        
        # Condition 4: Volume decreasing
        volume_sell_condition = dataframe['volume_ratio'] < 0.7
        conditions.append(volume_sell_condition)
        
        # Combine conditions
        dataframe['sell_signal'] = np.any(conditions, axis=0)
        
        return dataframe
