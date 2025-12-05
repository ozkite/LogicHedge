"""
Advanced perpetual trading system for Aster/Hyperliquid.
Features low-risk derivatives trading strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
import hashlib
import hmac
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class PerpetualContract:
    """Hyperliquid perpetual contract specification"""
    symbol: str
    base_asset: str
    quote_asset: str
    contract_size: float
    tick_size: float
    min_order_size: float
    max_leverage: float
    funding_interval: int  # hours
    max_position: float
    
@dataclass
class PerpetualPosition:
    """Perpetual position with risk metrics"""
    symbol: str
    side: str  # "long" or "short"
    size: float  # Contract size
    entry_price: float
    current_price: float
    leverage: float
    liquidation_price: float
    margin_used: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    funding_payments: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_funding_time: Optional[datetime] = None
    
    @property
    def margin_ratio(self) -> float:
        """Current margin ratio"""
        position_value = abs(self.size) * self.current_price
        if position_value == 0:
            return 0.0
        return (self.margin_used / position_value) * 100
        
    @property
    def distance_to_liquidation(self) -> float:
        """Percentage distance to liquidation"""
        if self.side == "long":
            distance = ((self.current_price - self.liquidation_price) / self.current_price) * 100
        else:
            distance = ((self.liquidation_price - self.current_price) / self.current_price) * 100
        return max(0.0, distance)
        
    @property
    def risk_score(self) -> float:
        """Risk score from 0 (safe) to 1 (dangerous)"""
        # Factors: margin ratio, distance to liquidation, leverage
        margin_risk = min(self.margin_ratio / 50, 1.0)  # 50% margin = max risk
        liquidation_risk = max(0, 1 - (self.distance_to_liquidation / 20))  # 20% buffer
        leverage_risk = min(self.leverage / 20, 1.0)  # 20x leverage = max risk
        
        return (margin_risk * 0.4 + liquidation_risk * 0.4 + leverage_risk * 0.2)

class AsterPerpetualTrader:
    """
    Low-risk perpetual trading system for Aster/Hyperliquid.
    Focuses on funding rate arbitrage, basis trading, and delta-neutral strategies.
    """
    
    def __init__(self, api_key: str, private_key: str, testnet: bool = True):
        self.api_key = api_key
        self.private_key = private_key
        self.testnet = testnet
        
        # Trading parameters
        self.max_leverage = 5.0
        self.max_position_size_usd = 10000.0
        self.target_margin_ratio = 0.3  # 30% margin
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.05  # 5%
        
        # State
        self.positions: Dict[str, PerpetualPosition] = {}
        self.funding_rates: Dict[str, float] = {}
        self.open_orders = {}
        self.account_balance = 0.0
        self.available_margin = 0.0
        
        # Risk limits
        self.daily_loss_limit = 500.0
        self.max_concurrent_positions = 3
        self.max_correlation = 0.7
        
        # Statistics
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Start maintenance tasks
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._track_funding_rates())
        
    async def initialize(self):
        """Initialize trading system"""
        # Load perpetual contracts
        await self._load_perpetual_contracts()
        
        # Get account balance
        await self._update_account_balance()
        
        logger.info("Aster perpetual trader initialized")
        
    async def _load_perpetual_contracts(self):
        """Load available perpetual contracts"""
        # These would come from API in production
        self.contracts = {
            "BTC-USDT": PerpetualContract(
                symbol="BTC-USDT",
                base_asset="BTC",
                quote_asset="USDT",
                contract_size=0.001,  # 0.001 BTC per contract
                tick_size=0.1,
                min_order_size=0.001,
                max_leverage=50.0,
                funding_interval=8,
                max_position=100.0  # Max 100 BTC
            ),
            "ETH-USDT": PerpetualContract(
                symbol="ETH-USDT",
                base_asset="ETH",
                quote_asset="USDT",
                contract_size=0.01,
                tick_size=0.01,
                min_order_size=0.01,
                max_leverage=50.0,
                funding_interval=8,
                max_position=1000.0
            ),
            "SOL-USDT": PerpetualContract(
                symbol="SOL-USDT",
                base_asset="SOL",
                quote_asset="USDT",
                contract_size=0.1,
                tick_size=0.001,
                min_order_size=0.1,
                max_leverage=25.0,
                funding_interval=8,
                max_position=10000.0
            )
        }
        
    async def execute_funding_rate_arbitrage(self, symbol: str):
        """
        Execute funding rate arbitrage strategy.
        Go long when funding is negative (shorts pay longs), short when positive.
        """
        contract = self.contracts.get(symbol)
        if not contract:
            return
            
        # Get current funding rate
        funding_rate = await self._get_funding_rate(symbol)
        if funding_rate is None:
            return
            
        self.funding_rates[symbol] = funding_rate
        
        # Calculate expected funding payment
        position_size = await self._calculate_optimal_position_size(symbol, funding_rate)
        
        if abs(funding_rate) >= 0.0005:  # 0.05% minimum
            if funding_rate < 0:
                # Negative funding: shorts pay longs → go long
                await self._open_position(symbol, "long", position_size)
                logger.info(f"Opening long on {symbol} for funding arbitrage: {funding_rate*100:.4f}%")
            else:
                # Positive funding: longs pay shorts → go short
                await self._open_position(symbol, "short", position_size)
                logger.info(f"Opening short on {symbol} for funding arbitrage: {funding_rate*100:.4f}%")
                
    async def _calculate_optimal_position_size(self, symbol: str, funding_rate: float) -> float:
        """Calculate optimal position size for funding arbitrage"""
        # Kelly Criterion for funding rate arb
        expected_return = abs(funding_rate) * 3 * 365  # Annualized (3 funding periods per day)
        win_prob = 0.85  # High probability for funding arb
        loss_prob = 0.15
        max_loss_pct = 0.02  # 2% stop loss
        
        kelly_fraction = (win_prob * expected_return - loss_prob * max_loss_pct) / expected_return
        kelly_fraction = max(0.05, min(kelly_fraction, 0.2))  # Cap at 20%
        
        available = min(self.available_margin * 0.5, self.max_position_size_usd)
        return available * kelly_fraction
        
    async def execute_basis_trading(self, symbol: str):
        """
        Basis trading: Trade the spread between perpetual and spot.
        Long perpetual + short spot when basis is high (contango).
        Short perpetual + long spot when basis is low (backwardation).
        """
        # Get perpetual price
        perp_price = await self._get_perpetual_price(symbol)
        
        # Get spot price (from CEX)
        spot_price = await self._get_spot_price(symbol)
        
        if perp_price is None or spot_price is None:
            return
            
        # Calculate basis
        basis = perp_price - spot_price
        basis_pct = (basis / spot_price) * 100
        
        # Calculate fair value basis (based on funding rates, time to expiry, etc.)
        fair_basis = await self._calculate_fair_basis(symbol)
        
        # Trading logic
        position_size = await self._calculate_basis_position_size(symbol, basis_pct)
        
        if basis_pct > fair_basis * 1.5:  # Basis too high
            # Short perpetual, long spot
            await self._open_position(symbol, "short", position_size)
            # Note: In production, you'd also take spot position on CEX
            logger.info(f"Opening basis trade: short {symbol} perpetual, basis={basis_pct:.2f}%")
            
        elif basis_pct < fair_basis * 0.5:  # Basis too low
            # Long perpetual, short spot
            await self._open_position(symbol, "long", position_size)
            logger.info(f"Opening basis trade: long {symbol} perpetual, basis={basis_pct:.2f}%")
            
    async def _calculate_fair_basis(self, symbol: str) -> float:
        """Calculate fair basis based on funding rates and risk-free rate"""
        # Simplified calculation
        funding_rate = self.funding_rates.get(symbol, 0.0)
        risk_free_rate = 0.05  # 5% annual
        
        # Fair basis = (risk_free_rate - funding_rate) * time_factor
        daily_rate = (risk_free_rate - funding_rate * 3) / 365  # 3 funding periods per day
        return daily_rate * 100  # Convert to percentage
        
    async def execute_delta_neutral_market_making(self, symbol: str):
        """
        Delta-neutral market making on perpetuals.
        Provide liquidity while hedging delta risk.
        """
        contract = self.contracts.get(symbol)
        if not contract:
            return
            
        # Get order book
        order_book = await self._get_order_book(symbol)
        if not order_book:
            return
            
        # Calculate fair price
        fair_price = await self._calculate_fair_price(symbol)
        
        # Place bids and asks around fair price
        spread_pct = 0.001  # 0.1% spread
        bid_price = fair_price * (1 - spread_pct / 2)
        ask_price = fair_price * (1 + spread_pct / 2)
        
        # Calculate order sizes based on inventory
        inventory = self._get_inventory(symbol)
        bid_size = await self._calculate_mm_order_size(symbol, "bid", inventory)
        ask_size = await self._calculate_mm_order_size(symbol, "ask", inventory)
        
        # Place orders
        await self._place_limit_order(symbol, "buy", bid_size, bid_price)
        await self._place_limit_order(symbol, "sell", ask_size, ask_price)
        
        # Hedge delta if needed
        await self._hedge_delta(symbol)
        
    async def _hedge_delta(self, symbol: str):
        """Hedge delta exposure"""
        contract = self.contracts.get(symbol)
        if not contract:
            return
            
        # Calculate net delta
        net_delta = 0.0
        for position in self.positions.values():
            if position.symbol == symbol:
                if position.side == "long":
                    net_delta += position.size
                else:
                    net_delta -= position.size
                    
        # Hedge if delta exceeds threshold
        if abs(net_delta) > contract.min_order_size * 10:
            # Hedge with spot or options
            # For now, just log
            logger.info(f"Delta hedging needed for {symbol}: {net_delta}")
            
    async def execute_low_risk_volatility_trading(self, symbol: str):
        """
        Low-risk volatility strategies:
        1. Strangle/straddle when IV is low
        2. Iron condor for range-bound markets
        3. Calendar spreads
        """
        # Get implied volatility
        iv = await self._get_implied_volatility(symbol)
        if iv is None:
            return
            
        # Get historical volatility
        hv = await self._get_historical_volatility(symbol, days=30)
        
        # Calculate volatility spread
        vol_spread = iv - hv
        
        if vol_spread > 0.05:  # IV > HV by 5%
            # Sell volatility (IV is rich)
            await self._sell_volatility(symbol, iv)
        elif vol_spread < -0.05:  # IV < HV by 5%
            # Buy volatility (IV is cheap)
            await self._buy_volatility(symbol, iv)
            
    async def _sell_volatility(self, symbol: str, iv: float):
        """Sell volatility through strangle/straddle"""
        # ATM strike
        atm_price = await self._get_atm_price(symbol)
        
        # Sell OTM call and put
        call_strike = atm_price * 1.05  # 5% OTM
        put_strike = atm_price * 0.95   # 5% OTM
        
        # Calculate position size based on vega exposure
        vega_exposure = await self._calculate_vega_exposure(symbol, iv)
        position_size = min(vega_exposure, self.max_position_size_usd * 0.1)
        
        # Note: Would place options orders here
        logger.info(f"Selling volatility on {symbol}: IV={iv*100:.1f}%, "
                  f"call@{call_strike:.0f}, put@{put_strike:.0f}")
                  
    async def _buy_volatility(self, symbol: str, iv: float):
        """Buy volatility through strangle/straddle"""
        atm_price = await self._get_atm_price(symbol)
        
        # Buy OTM call and put
        call_strike = atm_price * 1.10  # 10% OTM
        put_strike = atm_price * 0.90   # 10% OTM
        
        position_size = self.max_position_size_usd * 0.05  # Smaller size for buying
        
        logger.info(f"Buying volatility on {symbol}: IV={iv*100:.1f}%, "
                  f"call@{call_strike:.0f}, put@{put_strike:.0f}")
                  
    async def _open_position(self, symbol: str, side: str, size_usd: float):
        """Open a perpetual position"""
        contract = self.contracts.get(symbol)
        if not contract:
            return
            
        # Get current price
        price = await self._get_perpetual_price(symbol)
        if price is None:
            return
            
        # Calculate contract size
        contract_size = size_usd / price
        
        # Apply position limits
        contract_size = min(contract_size, contract.max_position)
        contract_size = max(contract_size, contract.min_order_size)
        
        # Calculate leverage (capped at max)
        leverage = min(self.max_leverage, contract.max_leverage)
        
        # Calculate margin required
        margin_required = (contract_size * price) / leverage
        
        # Check available margin
        if margin_required > self.available_margin * 0.8:  # 80% of available
            logger.warning(f"Insufficient margin for {symbol} position")
            return
            
        # Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price(
            price, side, leverage, self.target_margin_ratio
        )
        
        # Create position
        position = PerpetualPosition(
            symbol=symbol,
            side=side,
            size=contract_size if side == "long" else -contract_size,
            entry_price=price,
            current_price=price,
            leverage=leverage,
            liquidation_price=liquidation_price,
            margin_used=margin_required
        )
        
        # Update state
        self.positions[symbol] = position
        self.available_margin -= margin_required
        
        # Place order (simplified)
        order_result = await self._place_perpetual_order(symbol, side, contract_size, price)
        
        if order_result:
            logger.info(f"Opened {side} position on {symbol}: "
                      f"{contract_size:.4f} @ ${price:.2f}, "
                      f"Leverage: {leverage}x, Margin: ${margin_required:.2f}")
                      
        return position
        
    def _calculate_liquidation_price(self, entry_price: float, side: str, 
                                   leverage: float, margin_ratio: float) -> float:
        """Calculate liquidation price"""
        # Simplified formula
        if side == "long":
            return entry_price * (1 - (1 - margin_ratio) / leverage)
        else:
            return entry_price * (1 + (1 - margin_ratio) / leverage)
            
    async def _monitor_positions(self):
        """Monitor open positions for risk management"""
        while True:
            try:
                for symbol, position in list(self.positions.items()):
                    # Update price
                    current_price = await self._get_perpetual_price(symbol)
                    if current_price:
                        position.current_price = current_price
                        
                        # Update P&L
                        if position.side == "long":
                            position.unrealized_pnl = (
                                position.size * (current_price - position.entry_price)
                            )
                        else:
                            position.unrealized_pnl = (
                                abs(position.size) * (position.entry_price - current_price)
                            )
                            
                        # Check stop loss
                        if self._should_stop_loss(position):
                            await self._close_position(symbol, "stop_loss")
                            
                        # Check take profit
                        if self._should_take_profit(position):
                            await self._close_position(symbol, "take_profit")
                            
                        # Check margin call
                        if position.margin_ratio > 80:  # 80% margin ratio
                            logger.warning(f"Margin call warning for {symbol}: "
                                         f"{position.margin_ratio:.1f}%")
                            await self._reduce_position(symbol)
                            
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)
                
    def _should_stop_loss(self, position: PerpetualPosition) -> bool:
        """Check if stop loss should trigger"""
        pnl_pct = (position.unrealized_pnl / position.margin_used) * 100
        return pnl_pct <= -self.stop_loss_pct * 100
        
    def _should_take_profit(self, position: PerpetualPosition) -> bool:
        """Check if take profit should trigger"""
        pnl_pct = (position.unrealized_pnl / position.margin_used) * 100
        return pnl_pct >= self.take_profit_pct * 100
        
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        position = self.positions.get(symbol)
        if not position:
            return
            
        # Calculate exit price
        exit_price = position.current_price
        
        # Calculate final P&L
        if position.side == "long":
            realized_pnl = position.size * (exit_price - position.entry_price)
        else:
            realized_pnl = abs(position.size) * (position.entry_price - exit_price)
            
        # Update metrics
        self.total_pnl += realized_pnl
        self.daily_pnl += realized_pnl
        self.available_margin += position.margin_used
        
        # Record trade
        self.trade_history.append({
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": abs(position.size),
            "pnl": realized_pnl,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Close position
        del self.positions[symbol]
        
        logger.info(f"Closed {symbol} position: P&L=${realized_pnl:.2f}, Reason: {reason}")
        
    async def _reduce_position(self, symbol: str, reduce_by: float = 0.5):
        """Reduce position size (partial close)"""
        position = self.positions.get(symbol)
        if not position:
            return
            
        # Calculate size to close
        close_size = abs(position.size) * reduce_by
        
        # Partial close
        # Note: Implement actual partial close logic
        
        logger.info(f"Reducing {symbol} position by {reduce_by*100:.0f}%")
        
    async def _track_funding_rates(self):
        """Track and collect funding payments"""
        while True:
            try:
                for symbol in self.contracts.keys():
                    funding_rate = await self._get_funding_rate(symbol)
                    if funding_rate and symbol in self.positions:
                        position = self.positions[symbol]
                        
                        # Calculate funding payment
                        payment = position.size * position.current_price * funding_rate
                        position.funding_payments += payment
                        
                        if payment != 0:
                            logger.info(f"Funding payment for {symbol}: "
                                      f"${payment:.4f} ({funding_rate*100:.4f}%)")
                                          
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Funding tracking error: {e}")
                await asyncio.sleep(300)
                
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics"""
        total_margin = sum(p.margin_used for p in self.positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = self.total_pnl
        
        # Calculate risk metrics
        var_95 = self._calculate_var(confidence=0.95)
        expected_shortfall = self._calculate_expected_shortfall(confidence=0.95)
        
        return {
            "open_positions": len(self.positions),
            "total_margin_used": total_margin,
            "available_margin": self.available_margin,
            "unrealized_pnl": total_unrealized,
            "realized_pnl": total_realized,
            "total_pnl": total_unrealized + total_realized,
            "daily_pnl": self.daily_pnl,
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        # In production, use historical simulation or parametric methods
        pnl_values = [t["pnl"] for t in self.trade_history[-100:]]  # Last 100 trades
        if not pnl_values:
            return 0.0
            
        return np.percentile(pnl_values, (1 - confidence) * 100)
        
    def _calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        pnl_values = [t["pnl"] for t in self.trade_history[-100:]]
        if not pnl_values:
            return 0.0
            
        var = self._calculate_var(confidence)
        losses = [p for p in pnl_values if p <= var]
        
        if losses:
            return np.mean(losses)
        return var
        
    # Placeholder methods for actual API calls
    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate"""
        # Implement actual API call
        return 0.0001  # 0.01% placeholder
        
    async def _get_perpetual_price(self, symbol: str) -> Optional[float]:
        """Get perpetual price"""
        # Implement actual API call
        return 50000.0 if "BTC" in symbol else 3000.0
        
    async def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get spot price from CEX"""
        # Implement actual API call
        return 49950.0 if "BTC" in symbol else 2995.0
        
    async def _get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order book"""
        # Implement actual API call
        return {"bids": [[49900, 1.0]], "asks": [[50100, 1.0]]}
        
    async def _place_perpetual_order(self, symbol: str, side: str, size: float, 
                                    price: float) -> bool:
        """Place perpetual order"""
        # Implement actual API call
        return True
        
    async def _update_account_balance(self):
        """Update account balance"""
        # Implement actual API call
        self.account_balance = 50000.0
        self.available_margin = 50000.0
