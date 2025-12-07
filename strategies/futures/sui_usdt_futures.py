"""
SUI_USDT Futures Trading Strategy
Comprehensive strategy based on SUI-specific chart patterns and market behavior
"""

import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

@dataclass
class SUIChartPattern:
    """SUI-specific chart patterns"""
    name: str
    confidence: float
    pattern_type: str  # 'reversal', 'continuation', 'consolidation'
    timeframe: str
    detected_at: datetime

@dataclass
class FuturesSignal:
    side: PositionSide
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple TP levels
    leverage: int
    size: float  # Position size in contracts
    confidence: float
    pattern: Optional[SUIChartPattern] = None
    timeframe: str = "15m"
    reason: str = ""

@dataclass
class SUILevels:
    """Key SUI levels identified from chart"""
    resistance: List[float]
    support: List[float]
    pivot: float
    volume_profile_poc: float  # Point of Control
    high_volume_nodes: List[Tuple[float, float]]  # (price, volume)

class SUIUSDTFuturesStrategy:
    """
    SUI_USDT Futures Trading Strategy
    
    SUI Characteristics to consider:
    - High volatility (3-8% daily moves common)
    - Strong reaction to Bitcoin movements
    - Liquidity concentrated around round numbers
    - Frequent fakeouts before big moves
    - Volume spikes precede major moves
    """
    
    def __init__(self, 
                 exchange_client,
                 leverage: int = 10,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 use_isolated: bool = True):
        """
        Initialize SUI futures strategy
        
        Args:
            exchange_client: Exchange API client (Binance, Bybit, etc.)
            leverage: Trading leverage (1-125x)
            risk_per_trade: Risk per trade as percentage of capital
            use_isolated: Use isolated margin mode
        """
        self.exchange = exchange_client
        self.leverage = min(leverage, 125)  # Cap at 125x
        self.risk_per_trade = risk_per_trade
        self.use_isolated = use_isolated
        
        # SUI-specific parameters
        self.symbol = "SUI_USDT"
        self.futures_symbol = f"{self.symbol}:USDT"  # Adjust based on exchange
        
        # Multi-timeframe analysis
        self.timeframes = ["5m", "15m", "1h", "4h", "1d"]
        
        # Trading session times (SUI often moves during specific sessions)
        self.asian_session = (0, 8)  # UTC times
        self.london_session = (8, 16)
        self.us_session = (14, 22)
        
        # Initialize state
        self.current_position = None
        self.trade_history = []
        self.pnl = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        
        logger.info(f"Initialized SUI_USDT Futures Strategy with {leverage}x leverage")
    
    def analyze_multi_timeframe(self) -> Dict[str, pd.DataFrame]:
        """Analyze SUI across multiple timeframes"""
        analysis = {}
        
        for tf in self.timeframes:
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    self.futures_symbol, 
                    timeframe=tf, 
                    limit=100
                )
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add indicators
                df = self._add_indicators(df, tf)
                
                analysis[tf] = df
                
            except Exception as e:
                logger.error(f"Error fetching {tf} data: {e}")
                continue
        
        return analysis
    
    def _add_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add SUI-specific technical indicators"""
        
        # 1. Volume-based indicators (SUI reacts strongly to volume)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 2. Volatility indicators
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )
        
        # 3. Trend indicators
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # 4. Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # 5. SUI-specific: Reaction to round numbers
        df['round_number'] = (df['close'] // 0.1) * 0.1
        df['distance_to_round'] = df['close'] - df['round_number']
        
        # 6. Order flow (simplified)
        df['buy_pressure'] = self._calculate_buy_pressure(df)
        
        # 7. Market structure
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        return df
    
    def _calculate_buy_pressure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate buy/sell pressure based on price and volume"""
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # Positive price change with high volume = strong buying
        buy_pressure = np.where(
            (price_change > 0) & (volume_change > 0.5),
            1.0,
            np.where(
                (price_change < 0) & (volume_change > 0.5),
                -1.0,
                0.0
            )
        )
        
        return pd.Series(buy_pressure, index=df.index)
    
    def detect_sui_patterns(self, df: pd.DataFrame) -> List[SUIChartPattern]:
        """Detect SUI-specific chart patterns"""
        patterns = []
        
        # 1. SUI Fakeout Pattern (common before big moves)
        if len(df) >= 10:
            last_candles = df.iloc[-10:]
            
            # Check for fakeout above resistance
            if (last_candles['high'].iloc[-3] > last_candles['high'].iloc[-4] and
                last_candles['close'].iloc[-3] < last_candles['open'].iloc[-3] and
                last_candles['close'].iloc[-1] < last_candles['low'].iloc[-3]):
                patterns.append(SUIChartPattern(
                    name="SUI_Fakeout_Bearish",
                    confidence=0.7,
                    pattern_type="reversal",
                    timeframe="15m",
                    detected_at=datetime.now()
                ))
        
        # 2. SUI Volume Spike Pattern
        if df['volume_ratio'].iloc[-1] > 2.5:
            patterns.append(SUIChartPattern(
                name="SUI_Volume_Spike",
                confidence=0.8,
                pattern_type="continuation",
                timeframe="5m",
                detected_at=datetime.now()
            ))
        
        # 3. EMA Stack Pattern (SUI often respects EMA stacks)
        ema_order = all([
            df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1],
            df['ema_21'].iloc[-1] > df['ema_50'].iloc[-1],
            df['close'].iloc[-1] > df['ema_9'].iloc[-1]
        ])
        
        if ema_order:
            patterns.append(SUIChartPattern(
                name="SUI_EMA_Bull_Stack",
                confidence=0.75,
                pattern_type="continuation",
                timeframe="1h",
                detected_at=datetime.now()
            ))
        
        return patterns
    
    def identify_key_levels(self, df: pd.DataFrame) -> SUILevels:
        """Identify key SUI levels from chart"""
        
        # Use last 200 candles for level identification
        recent_df = df.iloc[-200:] if len(df) > 200 else df
        
        # 1. Recent highs and lows as resistance/support
        recent_highs = recent_df['high'].nlargest(5).tolist()
        recent_lows = recent_df['low'].nsmallest(5).tolist()
        
        # 2. Round number levels (SUI respects these)
        current_price = df['close'].iloc[-1]
        round_levels = []
        
        for i in range(-5, 6):
            level = round(current_price + i * 0.1, 1)
            round_levels.append(level)
        
        # 3. Pivot points
        pivot = self._calculate_pivot_points(df)
        
        # 4. Volume Profile (simplified)
        # Group prices into buckets and find high volume areas
        price_buckets = np.arange(
            recent_df['low'].min(), 
            recent_df['high'].max(), 
            0.01
        )
        
        volume_profile = []
        for i in range(len(price_buckets) - 1):
            bucket_mask = (recent_df['close'] >= price_buckets[i]) & \
                         (recent_df['close'] < price_buckets[i + 1])
            bucket_volume = recent_df.loc[bucket_mask, 'volume'].sum()
            volume_profile.append((price_buckets[i], bucket_volume))
        
        # Sort by volume and get high volume nodes
        volume_profile.sort(key=lambda x: x[1], reverse=True)
        high_volume_nodes = volume_profile[:10]
        
        poc_price = volume_profile[0][0] if volume_profile else current_price
        
        return SUILevels(
            resistance=sorted(list(set(recent_highs + round_levels))),
            support=sorted(list(set(recent_lows + round_levels))),
            pivot=pivot,
            volume_profile_poc=poc_price,
            high_volume_nodes=high_volume_nodes
        )
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> float:
        """Calculate classic pivot point"""
        if len(df) < 2:
            return df['close'].iloc[-1]
        
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        return pivot
    
    def generate_signals(self) -> Optional[FuturesSignal]:
        """Generate futures trading signals for SUI"""
        
        # Multi-timeframe analysis
        multi_tf_data = self.analyze_multi_timeframe()
        
        if not multi_tf_data:
            return None
        
        # Focus on 15m and 1h for primary signals
        df_15m = multi_tf_data.get('15m')
        df_1h = multi_tf_data.get('1h')
        
        if df_15m is None or df_1h is None:
            return None
        
        # Get current market data
        current_price = df_15m['close'].iloc[-1]
        current_rsi = df_15m['rsi'].iloc[-1]
        atr = df_15m['atr'].iloc[-1]
        
        # Detect patterns
        patterns = self.detect_sui_patterns(df_15m)
        
        # Identify key levels
        levels = self.identify_key_levels(df_15m)
        
        # Multi-timeframe alignment check
        trend_alignment = self._check_trend_alignment(multi_tf_data)
        
        # Generate signal based on conditions
        signal = self._evaluate_trading_conditions(
            df_15m, df_1h, current_price, current_rsi, atr, 
            patterns, levels, trend_alignment
        )
        
        return signal
    
    def _check_trend_alignment(self, multi_tf_data: Dict) -> Dict:
        """Check if trends align across timeframes"""
        alignment = {
            'bullish_aligned': False,
            'bearish_aligned': False,
            'alignment_score': 0.0
        }
        
        bullish_count = 0
        total_tf = 0
        
        for tf, df in multi_tf_data.items():
            if len(df) < 2:
                continue
            
            total_tf += 1
            current_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            
            # Simple trend detection
            if current_close > df['ema_21'].iloc[-1]:
                bullish_count += 1
            elif current_close < df['ema_21'].iloc[-1]:
                bullish_count -= 1
        
        alignment_score = bullish_count / total_tf if total_tf > 0 else 0
        
        if alignment_score >= 0.7:
            alignment['bullish_aligned'] = True
        elif alignment_score <= -0.7:
            alignment['bearish_aligned'] = True
        
        alignment['alignment_score'] = alignment_score
        
        return alignment
    
    def _evaluate_trading_conditions(self, 
                                    df_15m: pd.DataFrame,
                                    df_1h: pd.DataFrame,
                                    current_price: float,
                                    current_rsi: float,
                                    atr: float,
                                    patterns: List[SUIChartPattern],
                                    levels: SUILevels,
                                    trend_alignment: Dict) -> Optional[FuturesSignal]:
        """Evaluate all conditions and generate signal"""
        
        # Risk management first
        if not self._risk_management_check():
            return None
        
        # Check trading session (SUI often moves during specific sessions)
        current_hour = datetime.now().hour
        
        # SUI is often most active during US and Asian sessions overlap
        optimal_session = (current_hour >= 14 and current_hour <= 22) or \
                         (current_hour >= 0 and current_hour <= 8)
        
        if not optimal_session:
            logger.info("Not optimal trading session for SUI")
            return None
        
        # Signal generation logic
        
        # 1. Trend following with pullback (High probability setup)
        if trend_alignment['bullish_aligned']:
            # Look for pullback to support
            nearest_support = max([s for s in levels.support if s < current_price], default=None)
            
            if (nearest_support and 
                abs(current_price - nearest_support) < atr * 0.5 and
                current_rsi < 45 and
                df_15m['buy_pressure'].iloc[-1] > 0):
                
                stop_loss = nearest_support - atr * 0.5
                risk = current_price - stop_loss
                
                # Calculate position size
                position_size = self._calculate_position_size(current_price, stop_loss)
                
                # Multiple take profit levels (SUI often hits multiple targets)
                take_profit = [
                    current_price + risk * 1.5,  # TP1
                    current_price + risk * 2.5,  # TP2
                    current_price + risk * 4.0   # TP3
                ]
                
                return FuturesSignal(
                    side=PositionSide.LONG,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self.leverage,
                    size=position_size,
                    confidence=0.75,
                    patterns=patterns if patterns else None,
                    timeframe="15m",
                    reason="Bullish trend alignment with pullback to support"
                )
        
        # 2. Mean reversion at extremes (SUI often reverses at RSI extremes)
        if current_rsi > 75 and df_15m['volume_ratio'].iloc[-1] > 2:
            # Overbought with high volume -> potential reversal
            
            nearest_resistance = min([r for r in levels.resistance if r > current_price], default=None)
            
            if nearest_resistance:
                stop_loss = nearest_resistance + atr * 0.5
                risk = stop_loss - current_price
                
                position_size = self._calculate_position_size(current_price, stop_loss, is_short=True)
                
                take_profit = [
                    current_price - risk * 1.0,
                    current_price - risk * 2.0,
                    current_price - risk * 3.0
                ]
                
                return FuturesSignal(
                    side=PositionSide.SHORT,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self.leverage,
                    size=position_size,
                    confidence=0.65,
                    patterns=patterns if patterns else None,
                    timeframe="15m",
                    reason="Overbought reversal with volume confirmation"
                )
        
        # 3. Breakout strategy (SUI often has clean breakouts)
        if df_15m['volume_ratio'].iloc[-1] > 3.0:
            # High volume breakout
            
            # Check if breaking above resistance
            for resistance in levels.resistance:
                if (current_price > resistance and 
                    df_15m['close'].iloc[-2] < resistance and
                    df_15m['volume'].iloc[-1] > df_15m['volume'].iloc[-2] * 1.5):
                    
                    stop_loss = resistance - atr
                    position_size = self._calculate_position_size(current_price, stop_loss)
                    
                    take_profit = [
                        current_price + atr * 1.5,
                        current_price + atr * 2.5,
                        current_price + atr * 4.0
                    ]
                    
                    return FuturesSignal(
                        side=PositionSide.LONG,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        leverage=self.leverage,
                        size=position_size,
                        confidence=0.7,
                        patterns=patterns if patterns else None,
                        timeframe="15m",
                        reason="Volume breakout above key resistance"
                    )
        
        return None
    
    def _calculate_position_size(self, 
                                entry_price: float, 
                                stop_loss: float,
                                is_short: bool = False) -> float:
        """Calculate position size based on risk management"""
        
        # Calculate risk per trade in USD
        risk_amount = self._get_account_balance() * self.risk_per_trade
        
        # Calculate price risk
        if is_short:
            price_risk = stop_loss - entry_price  # For short positions
        else:
            price_risk = entry_price - stop_loss  # For long positions
        
        if price_risk <= 0:
            return 0
        
        # Calculate position size in contracts
        # Risk per contract = price_risk * contract_multiplier
        # For crypto futures, typically 1 contract = 1 unit of underlying
        
        position_size = risk_amount / price_risk
        
        # Adjust for leverage
        position_size = position_size * self.leverage
        
        # Cap position size to reasonable limits
        max_position = self._get_account_balance() * 0.1 * self.leverage
        
        return min(position_size, max_position)
    
    def _get_account_balance(self) -> float:
        """Get available balance for trading"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return 1000  # Default for testing
    
    def _risk_management_check(self) -> bool:
        """Check if risk limits allow new trades"""
        
        # Check daily loss limit
        if self.pnl < -self._get_account_balance() * 0.05:  # 5% daily loss limit
            logger.warning("Daily loss limit reached")
            return False
        
        # Check consecutive losses
        if len(self.trade_history) >= 3:
            recent_losses = sum(1 for trade in self.trade_history[-3:] if trade['pnl'] < 0)
            if recent_losses >= 3:
                logger.warning("3 consecutive losses - taking break")
                return False
        
        # Check if already in position
        if self.current_position is not None:
            logger.info("Already in position")
            return False
        
        return True
    
    def execute_trade(self, signal: FuturesSignal) -> Dict:
        """Execute the futures trade"""
        
        try:
            trade_result = {
                'signal': asdict(signal),
                'execution_time': datetime.now(),
                'status': 'pending',
                'orders': []
            }
            
            # 1. Set leverage
            self.exchange.set_leverage(
                symbol=self.futures_symbol,
                leverage=self.leverage
            )
            
            # 2. Set margin mode
            if self.use_isolated:
                self.exchange.set_margin_mode(
                    symbol=self.futures_symbol,
                    margin_mode='isolated'
                )
            
            # 3. Place main order
            side = 'buy' if signal.side == PositionSide.LONG else 'sell'
            
            main_order = self.exchange.create_order(
                symbol=self.futures_symbol,
                type='limit',
                side=side,
                amount=signal.size,
                price=signal.entry_price
            )
            
            trade_result['orders'].append(main_order)
            
            # 4. Place stop loss
            stop_order = self.exchange.create_order(
                symbol=self.futures_symbol,
                type='stop_market',
                side='sell' if signal.side == PositionSide.LONG else 'buy',
                amount=signal.size,
                params={
                    'stopPrice': signal.stop_loss,
                    'reduceOnly': True
                }
            )
            
            trade_result['orders'].append(stop_order)
            
            # 5. Place take profit orders (bracket orders)
            for i, tp_price in enumerate(signal.take_profit):
                tp_order = self.exchange.create_order(
                    symbol=self.futures_symbol,
                    type='take_profit_market',
                    side='sell' if signal.side == PositionSide.LONG else 'buy',
                    amount=signal.size / len(signal.take_profit) if i == 0 else signal.size,
                    params={
                        'stopPrice': tp_price,
                        'reduceOnly': True
                    }
                )
                trade_result['orders'].append(tp_order)
            
            trade_result['status'] = 'executed'
            
            # Update position
            self.current_position = {
                'side': signal.side.value,
                'entry_price': signal.entry_price,
                'size': signal.size,
                'stop_loss': signal.stop_loss,
                'take_profits': signal.take_profit,
                'entry_time': datetime.now()
            }
            
            logger.info(f"Trade executed: {signal.side.value} {signal.size} contracts @ {signal.entry_price}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def monitor_position(self) -> Dict:
        """Monitor current position and manage exits"""
        
        if self.current_position is None:
            return {'status': 'no_position'}
        
        try:
            # Get current market price
            ticker = self.exchange.fetch_ticker(self.futures_symbol)
            current_price = ticker['last']
            
            position = self.current_position
            
            # Calculate unrealized PnL
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            # Check for partial profit taking
            for i, tp in enumerate(position['take_profits']):
                if ((position['side'] == 'LONG' and current_price >= tp) or
                    (position['side'] == 'SHORT' and current_price <= tp)):
                    
                    # Close portion of position
                    close_amount = position['size'] / len(position['take_profits'])
                    
                    close_order = self.exchange.create_order(
                        symbol=self.futures_symbol,
                        type='market',
                        side='sell' if position['side'] == 'LONG' else 'buy',
                        amount=close_amount,
                        params={'reduceOnly': True}
                    )
                    
                    logger.info(f"Partial TP {i+1} hit at {tp}, closed {close_amount} contracts")
                    
                    # Update position size
                    position['size'] -= close_amount
            
            # Check for trailing stop (if implemented)
            # Update trailing stop based on ATR or percentage
            
            return {
                'status': 'monitoring',
                'current_price': current_price,
                'unrealized_pnl_pct': pnl_pct,
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_position(self, reason: str = "manual_close") -> Dict:
        """Close current position"""
        
        if self.current_position is None:
            return {'status': 'no_position'}
        
        try:
            position = self.current_position
            
            # Calculate exit price
            ticker = self.exchange.fetch_ticker(self.futures_symbol)
            exit_price = ticker['last']
            
            # Close position
            close_side = 'sell' if position['side'] == 'LONG' else 'buy'
            
            close_order = self.exchange.create_order(
                symbol=self.futures_symbol,
                type='market',
                side=close_side,
                amount=position['size'],
                params={'reduceOnly': True}
            )
            
            # Calculate realized PnL
            if position['side'] == 'LONG':
                realized_pnl = (exit_price - position['entry_price']) * position['size']
            else:
                realized_pnl = (position['entry_price'] - exit_price) * position['size']
            
            # Update statistics
            self.pnl += realized_pnl
            self.trade_history.append({
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'size': position['size'],
                'pnl': realized_pnl,
                'reason': reason,
                'timestamp': datetime.now()
            })
            
            # Update win rate
            wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            self.win_rate = wins / len(self.trade_history) if self.trade_history else 0
            
            logger.info(f"Position closed: PnL = {realized_pnl:.2f} USDT, Reason: {reason}")
            
            # Clear current position
            self.current_position = None
            
            return {
                'status': 'closed',
                'exit_price': exit_price,
                'realized_pnl': realized_pnl,
                'order': close_order
            }
            
        except Exception as e:
            logger.error(f"Position close failed: {e}")
            return {'status': 'error', 'error': str(e)}
