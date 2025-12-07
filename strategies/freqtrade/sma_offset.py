"""
SMAOffset strategy from Freqtrade
Simple moving average with offset for high-frequency trading
"""

class SMAOffsetStrategy:
    """
    SMAOffset strategy: Buy when price crosses above SMA + offset
    """
    
    def __init__(self):
        self.name = "SMAOffset"
        self.config = {
            'sma_short': 10,
            'sma_long': 30,
            'offset_pct': 0.002,  # 0.2% offset
            'volume_threshold': 1.5,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'trailing_stop': -0.02,
            'stop_loss': -0.05
        }
        
    async def calculate_indicators(self, dataframe):
        """Calculate required indicators"""
        # SMAs
        dataframe['sma_short'] = talib.SMA(dataframe['close'], timeperiod=self.config['sma_short'])
        dataframe['sma_long'] = talib.SMA(dataframe['close'], timeperiod=self.config['sma_long'])
        
        # SMA with offset
        dataframe['sma_offset'] = dataframe['sma_short'] * (1 + self.config['offset_pct'])
        
        # RSI
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=self.config['rsi_period'])
        
        # Volume indicators
        dataframe['volume_sma'] = talib.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Price distance from SMA
        dataframe['price_distance'] = (dataframe['close'] - dataframe['sma_short']) / dataframe['sma_short']
        
        return dataframe
        
    async def generate_buy_signals(self, dataframe):
        """Generate buy signals"""
        conditions = []
        
        # Condition 1: Price crosses above SMA + offset
        price_condition = (
            (dataframe['close'] > dataframe['sma_offset']) &
            (dataframe['close'].shift(1) <= dataframe['sma_offset'].shift(1))
        )
        conditions.append(price_condition)
        
        # Condition 2: Short SMA above Long SMA (trend)
        trend_condition = dataframe['sma_short'] > dataframe['sma_long']
        conditions.append(trend_condition)
        
        # Condition 3: Volume spike
        volume_condition = dataframe['volume_ratio'] > self.config['volume_threshold']
        conditions.append(volume_condition)
        
        # Condition 4: RSI not overbought
        rsi_condition = dataframe['rsi'] < self.config['rsi_overbought']
        conditions.append(rsi_condition)
        
        # Combine conditions
        dataframe['buy_signal'] = np.all(conditions, axis=0)
        
        # Calculate buy price (SMA offset as target)
        dataframe['buy_price'] = dataframe['sma_offset']
        
        return dataframe
        
    async def generate_sell_signals(self, dataframe):
        """Generate sell signals"""
        conditions = []
        
        # Condition 1: Price crosses below SMA
        price_condition = (
            (dataframe['close'] < dataframe['sma_short']) &
            (dataframe['close'].shift(1) >= dataframe['sma_short'].shift(1))
        )
        conditions.append(price_condition)
        
        # Condition 2: RSI overbought
        rsi_condition = dataframe['rsi'] > self.config['rsi_overbought']
        conditions.append(rsi_condition)
        
        # Condition 3: Volume decreasing
        volume_condition = dataframe['volume_ratio'] < 0.8
        conditions.append(volume_condition)
        
        # Combine conditions
        dataframe['sell_signal'] = np.any(conditions, axis=0)
        
        return dataframe
        
    async def calculate_position_size(self, capital: float, current_price: float, 
                                    risk_pct: float = 0.02) -> float:
        """Calculate position size based on risk"""
        stop_loss_distance = abs(self.config['stop_loss'])
        risk_amount = capital * risk_pct
        
        position_size = risk_amount / (current_price * stop_loss_distance)
        
        return position_size
