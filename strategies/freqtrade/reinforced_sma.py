"""
ReinforcedSMA strategy with adaptive parameters
Machine learning enhanced SMA strategy
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ReinforcedSMAStrategy:
    """
    SMA strategy reinforced with machine learning
    Adapts parameters based on market conditions
    """
    
    def __init__(self):
        self.name = "ReinforcedSMA"
        
        # Base parameters
        self.base_config = {
            'sma_short': 10,
            'sma_long': 30,
            'sma_signal': 5,
            'rsi_period': 14,
            'atr_period': 14,
            'volume_period': 20
        }
        
        # ML model for parameter optimization
        self.ml_model = None
        self.training_data = []
        
        # Market regime detection
        self.regime = 'neutral'  # bullish, bearish, neutral
        
    async def detect_market_regime(self, dataframe):
        """Detect current market regime"""
        # Calculate trend indicators
        returns = dataframe['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # SMA slope
        sma_short = talib.SMA(dataframe['close'], timeperiod=self.base_config['sma_short'])
        sma_long = talib.SMA(dataframe['close'], timeperiod=self.base_config['sma_long'])
        
        # Determine regime
        if sma_short.iloc[-1] > sma_long.iloc[-1] and volatility.iloc[-1] < 0.02:
            self.regime = 'bullish'
        elif sma_short.iloc[-1] < sma_long.iloc[-1] and volatility.iloc[-1] < 0.02:
            self.regime = 'bearish'
        elif volatility.iloc[-1] > 0.03:
            self.regime = 'high_volatility'
        else:
            self.regime = 'neutral'
            
        return self.regime
        
    async def adapt_parameters(self, regime: str):
        """Adapt parameters based on market regime"""
        adapted_config = self.base_config.copy()
        
        if regime == 'bullish':
            # More aggressive in bull markets
            adapted_config['sma_short'] = 8
            adapted_config['sma_long'] = 25
            adapted_config['sma_signal'] = 3
            
        elif regime == 'bearish':
            # More conservative in bear markets
            adapted_config['sma_short'] = 15
            adapted_config['sma_long'] = 40
            adapted_config['sma_signal'] = 8
            
        elif regime == 'high_volatility':
            # Wider parameters for high volatility
            adapted_config['sma_short'] = 20
            adapted_config['sma_long'] = 50
            adapted_config['sma_signal'] = 10
            
        return adapted_config
        
    async def train_ml_model(self, historical_data):
        """Train ML model to optimize parameters"""
        # Prepare features
        features = []
        labels = []
        
        for i in range(len(historical_data) - 100):
            window = historical_data.iloc[i:i+100]
            
            # Calculate features
            sma_short = talib.SMA(window['close'], timeperiod=10)
            sma_long = talib.SMA(window['close'], timeperiod=30)
            rsi = talib.RSI(window['close'], timeperiod=14)
            volume_ratio = window['volume'] / talib.SMA(window['volume'], timeperiod=20)
            
            # Feature vector
            feature_vector = [
                (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1],
                rsi.iloc[-1],
                volume_ratio.iloc[-1],
                window['close'].pct_change().std()
            ]
            
            # Label: 1 if next candle is profitable, 0 otherwise
            future_return = (historical_data['close'].iloc[i+101] - 
                           historical_data['close'].iloc[i+100]) / historical_data['close'].iloc[i+100]
            label = 1 if future_return > 0.001 else 0
            
            features.append(feature_vector)
            labels.append(label)
            
        # Train model
        if features and labels:
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(features, labels)
            
    async def ml_predict_signal(self, dataframe):
        """Use ML model to predict buy/sell signals"""
        if not self.ml_model:
            return None
            
        # Calculate features for current data
        sma_short = talib.SMA(dataframe['close'], timeperiod=10)
        sma_long = talib.SMA(dataframe['close'], timeperiod=30)
        rsi = talib.RSI(dataframe['close'], timeperiod=14)
        volume_ratio = dataframe['volume'] / talib.SMA(dataframe['volume'], timeperiod=20)
        
        features = np.array([[
            (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1],
            rsi.iloc[-1],
            volume_ratio.iloc[-1],
            dataframe['close'].pct_change().std()
        ]])
        
        # Predict
        prediction = self.ml_model.predict(features)
        probability = self.ml_model.predict_proba(features)[0][1]
        
        return {
            'signal': prediction[0],
            'confidence': probability,
            'features': features[0].tolist()
        }
        
    async def generate_signals(self, dataframe, capital: float = 10000):
        """Generate trading signals with adaptive parameters"""
        # Detect market regime
        regime = await self.detect_market_regime(dataframe)
        
        # Adapt parameters
        config = await self.adapt_parameters(regime)
        
        # Calculate indicators with adapted parameters
        dataframe['sma_short'] = talib.SMA(dataframe['close'], timeperiod=config['sma_short'])
        dataframe['sma_long'] = talib.SMA(dataframe['close'], timeperiod=config['sma_long'])
        dataframe['sma_signal'] = talib.SMA(dataframe['close'], timeperiod=config['sma_signal'])
        
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=config['rsi_period'])
        dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'],
                                   timeperiod=config['atr_period'])
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(dataframe)):
            # Buy conditions
            buy_condition = (
                (dataframe['sma_short'].iloc[i] > dataframe['sma_long'].iloc[i]) &
                (dataframe['sma_short'].iloc[i-1] <= dataframe['sma_long'].iloc[i-1]) &
                (dataframe['rsi'].iloc[i] < 70) &
                (dataframe['volume'].iloc[i] > dataframe['volume'].rolling(window=20).mean().iloc[i])
            )
            
            # Sell conditions
            sell_condition = (
                (dataframe['sma_short'].iloc[i] < dataframe['sma_signal'].iloc[i]) &
                (dataframe['sma_short'].iloc[i-1] >= dataframe['sma_signal'].iloc[i-1]) |
                (dataframe['rsi'].iloc[i] > 80)
            )
            
            buy_signals.append(buy_condition)
            sell_signals.append(sell_condition)
            
        dataframe['buy_signal'] = buy_signals
        dataframe['sell_signal'] = sell_signals
        
        # Add ML prediction if available
        ml_prediction = await self.ml_predict_signal(dataframe)
        if ml_prediction:
            dataframe['ml_signal'] = ml_prediction['signal']
            dataframe['ml_confidence'] = ml_prediction['confidence']
            
            # Combine ML signal with technical signals
            dataframe['final_buy_signal'] = (
                dataframe['buy_signal'] & 
                (dataframe['ml_signal'] == 1) &
                (dataframe['ml_confidence'] > 0.7)
            )
        else:
            dataframe['final_buy_signal'] = dataframe['buy_signal']
            
        return dataframe
