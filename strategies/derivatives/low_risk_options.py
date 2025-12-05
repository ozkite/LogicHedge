"""
Low-risk options trading strategies for perpetuals:
- Covered calls
- Cash-secured puts
- Iron condors
- Calendar spreads
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Options contract specification"""
    symbol: str
    strike: float
    expiration: datetime
    option_type: str  # "call" or "put"
    iv: float  # Implied volatility
    delta: float
    gamma: float
    theta: float
    vega: float
    bid: float
    ask: float
    volume: int
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
        
    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - datetime.utcnow()).days

@dataclass
class OptionsPosition:
    """Options trading position"""
    position_id: str
    strategy: str
    legs: List[Dict[str, Any]]  # [{option: OptionContract, quantity: 1, side: "buy"}]
    entry_price: float
    current_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    entry_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self._total_quantity()
        
    def _total_quantity(self) -> int:
        return sum(abs(leg["quantity"]) for leg in self.legs)

class LowRiskOptionsTrader:
    """
    Low-risk options trading strategies for derivatives.
    Focuses on positive theta (time decay) strategies.
    """
    
    def __init__(self, capital: float = 100000.0):
        self.capital = capital
        self.allocated_capital = 0.0
        self.positions: Dict[str, OptionsPosition] = {}
        
        # Risk parameters
        self.max_capital_per_trade = 0.1  # 10% of capital
        self.max_delta_exposure = 0.2  # Max 20% delta
        self.max_vega_exposure = 10000.0  # Max $10k vega
        self.min_credit_received = 0.01  # Min 1% credit
        self.max_risk_reward_ratio = 0.33  # Risk 1 to make 3
        
        # Strategy parameters
        self.target_theta = 0.01  # Target 1% daily theta
        self.min_days_to_expiry = 7
        self.max_days_to_expiry = 45
        
    async def execute_covered_call(self, underlying_symbol: str, 
                                  spot_position_size: float):
        """
        Covered call strategy:
        - Own underlying asset
        - Sell call options against it
        Generates income while limiting upside.
        """
        # Get ATM call option
        underlying_price = await self._get_underlying_price(underlying_symbol)
        atm_strike = self._round_to_strike(underlying_price)
        
        options = await self._get_options_chain(underlying_symbol)
        call_option = self._find_option(options, atm_strike, "call", 30)  # 30 DTE
        
        if not call_option:
            return
            
        # Calculate position size
        max_contracts = int(spot_position_size / 100)  # Assuming 100 shares per contract
        contracts_to_sell = min(max_contracts, 10)  # Max 10 contracts
        
        # Calculate credit received
        credit_received = call_option.mid_price * contracts_to_sell * 100
        
        # Calculate max profit and loss
        max_profit = credit_received + ((call_option.strike - underlying_price) * 
                                      contracts_to_sell * 100)
        max_loss = underlying_price * contracts_to_sell * 100  # If underlying goes to 0
        
        # Risk-reward check
        if credit_received / max_loss < self.min_credit_received:
            return
            
        # Create position
        position = OptionsPosition(
            position_id=str(uuid.uuid4()),
            strategy="covered_call",
            legs=[
                {
                    "option": call_option,
                    "quantity": -contracts_to_sell,  # Negative = sell
                    "side": "sell"
                }
            ],
            entry_price=call_option.mid_price,
            current_price=call_option.mid_price,
            delta=call_option.delta * contracts_to_sell * 100,
            gamma=call_option.gamma * contracts_to_sell * 100,
            theta=call_option.theta * contracts_to_sell * 100,
            vega=call_option.vega * contracts_to_sell * 100,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[underlying_price - (credit_received / (contracts_to_sell * 100))]
        )
        
        self.positions[position.position_id] = position
        self.allocated_capital += max_loss
        
        logger.info(f"Executed covered call: {underlying_symbol} {call_option.strike}C "
                  f"Credit: ${credit_received:.2f}, Max Profit: ${max_profit:.2f}")
                  
        return position
        
    async def execute_cash_secured_put(self, underlying_symbol: str, 
                                      strike_price: Optional[float] = None):
        """
        Cash-secured put strategy:
        - Sell put option
        - Keep cash to buy if assigned
        Collects premium while willing to buy at lower price.
        """
        underlying_price = await self._get_underlying_price(underlying_symbol)
        
        if strike_price is None:
            strike_price = underlying_price * 0.95  # 5% OTM
            
        strike_price = self._round_to_strike(strike_price)
        
        options = await self._get_options_chain(underlying_symbol)
        put_option = self._find_option(options, strike_price, "put", 45)  # 45 DTE
        
        if not put_option:
            return
            
        # Calculate position size
        max_contracts = int((self.capital * 0.1) / (strike_price * 100))  # 10% of capital
        contracts_to_sell = min(max_contracts, 5)
        
        # Calculate credit and margin required
        credit_received = put_option.mid_price * contracts_to_sell * 100
        margin_required = strike_price * contracts_to_sell * 100
        
        # Risk-reward check
        risk_reward_ratio = margin_required / credit_received
        if risk_reward_ratio > 1 / self.max_risk_reward_ratio:
            return
            
        position = OptionsPosition(
            position_id=str(uuid.uuid4()),
            strategy="cash_secured_put",
            legs=[
                {
                    "option": put_option,
                    "quantity": -contracts_to_sell,
                    "side": "sell"
                }
            ],
            entry_price=put_option.mid_price,
            current_price=put_option.mid_price,
            delta=put_option.delta * contracts_to_sell * 100,
            gamma=put_option.gamma * contracts_to_sell * 100,
            theta=put_option.theta * contracts_to_sell * 100,
            vega=put_option.vega * contracts_to_sell * 100,
            max_profit=credit_received,
            max_loss=margin_required - credit_received,
            breakeven_points=[strike_price - put_option.mid_price]
        )
        
        self.positions[position.position_id] = position
        self.allocated_capital += margin_required
        
        logger.info(f"Executed cash-secured put: {underlying_symbol} {put_option.strike}P "
                  f"Credit: ${credit_received:.2f}, Breakeven: ${position.breakeven_points[0]:.2f}")
                  
        return position
        
    async def execute_iron_condor(self, underlying_symbol: str, width_pct: float = 0.1):
        """
        Iron condor strategy:
        - Sell OTM call spread + OTM put spread
        - Collect premium with defined risk
        Best for range-bound markets.
        """
        underlying_price = await self._get_underlying_price(underlying_symbol)
        
        # Calculate strikes
        short_put_strike = underlying_price * (1 - width_pct/2)
        long_put_strike = short_put_strike * (1 - width_pct)
        short_call_strike = underlying_price * (1 + width_pct/2)
        long_call_strike = short_call_strike * (1 + width_pct)
        
        # Round strikes
        strikes = [self._round_to_strike(s) for s in 
                  [long_put_strike, short_put_strike, short_call_strike, long_call_strike]]
        
        # Get options
        options = await self._get_options_chain(underlying_symbol)
        expiry = self._find_optimal_expiry(options)
        
        long_put = self._find_option(options, strikes[0], "put", expiry)
        short_put = self._find_option(options, strikes[1], "put", expiry)
        short_call = self._find_option(options, strikes[2], "call", expiry)
        long_call = self._find_option(options, strikes[3], "call", expiry)
        
        if not all([long_put, short_put, short_call, long_call]):
            return
            
        # Calculate credit received
        credit = (short_put.mid_price + short_call.mid_price - 
                 long_put.mid_price - long_call.mid_price)
        
        if credit <= 0:
            return
            
        # Calculate max risk
        risk = (short_call.strike - short_put.strike) - credit
        
        # Check risk-reward
        if credit / risk < self.max_risk_reward_ratio:
            return
            
        position = OptionsPosition(
            position_id=str(uuid.uuid4()),
            strategy="iron_condor",
            legs=[
                {"option": long_put, "quantity": 1, "side": "buy"},
                {"option": short_put, "quantity
