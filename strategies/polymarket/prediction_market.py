"""
Polymarket Prediction Market Trading Strategy
Arbitrage, market making, and information edge strategies for prediction markets
"""

import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import logging
from web3 import Web3
from eth_abi import abi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketType(Enum):
    BINARY = "binary"
    CATEGORICAL = "categorical"
    SCALAR = "scalar"

class PositionType(Enum):
    YES = "yes"
    NO = "no"
    LONG = "long"  # For scalar markets
    SHORT = "short"

@dataclass
class PolymarketMarket:
    """Polymarket market data structure"""
    market_id: str
    question: str
    description: str
    market_type: MarketType
    outcomes: List[str]
    end_time: int
    volume: Decimal
    liquidity: Decimal
    last_price: Decimal
    condition_id: str
    resolution_source: str
    active: bool

@dataclass
class OrderBook:
    """Polymarket order book for a specific outcome"""
    outcome: str
    bids: List[Tuple[Decimal, Decimal]]  # (price, amount)
    asks: List[Tuple[Decimal, Decimal]]
    last_trade: Optional[Tuple[Decimal, Decimal]] = None  # (price, amount)

@dataclass
class TradingSignal:
    """Trading signal for Polymarket"""
    market_id: str
    outcome: str
    position_type: PositionType
    amount: Decimal
    price: Decimal
    confidence: float
    strategy: str
    expected_value: float = 0.0
    expected_profit: float = 0.0
    kelly_fraction: float = 0.0

class PolymarketTradingBot:
    """
    Main Polymarket trading bot implementing various strategies
    
    Sources integrated:
    - https://github.com/Polymarket/agents (official agents)
    - https://github.com/Trust412/Polymarket-spike-bot-v1 (spike detection)
    - https://github.com/dappboris-dev/polymarket-trading-bot (trading bot)
    - https://github.com/Trust412/polymarket-copy-trading-bot-v1 (copy trading)
    """
    
    def __init__(self, 
                 web3_provider: str,
                 private_key: str = None,
                 polymarket_address: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
                 conditional_tokens_address: str = "0xC59b0e4De5F1248C1140964E0fF287B192407E0C",
                 gas_limit: int = 500000):
        """
        Initialize Polymarket trading bot
        
        Args:
            web3_provider: Web3 provider URL
            private_key: Private key for trading (optional for read-only)
            polymarket_address: Polymarket contract address
            conditional_tokens_address: Conditional tokens contract address
            gas_limit: Gas limit for transactions
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Contract addresses
        self.polymarket_address = polymarket_address
        self.conditional_tokens_address = conditional_tokens_address
        
        # Load ABI
        with open("abis/polymarket_abi.json", "r") as f:
            polymarket_abi = json.load(f)
        
        with open("abis/conditional_tokens_abi.json", "r") as f:
            conditional_tokens_abi = json.load(f)
        
        # Initialize contracts
        self.polymarket = self.w3.eth.contract(
            address=self.w3.to_checksum_address(polymarket_address),
            abi=polymarket_abi
        )
        
        self.conditional_tokens = self.w3.eth.contract(
            address=self.w3.to_checksum_address(conditional_tokens_address),
            abi=conditional_tokens_abi
        )
        
        # Trading parameters
        self.gas_limit = gas_limit
        self.gas_price_multiplier = 1.2
        self.min_profit_threshold = Decimal('0.02')  # 2% minimum profit
        self.max_position_size = Decimal('1000')  # Max position in USDC
        
        # State
        self.active_markets: Dict[str, PolymarketMarket] = {}
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        logger.info("Polymarket Trading Bot initialized")
    
    async def fetch_active_markets(self, limit: int = 50) -> List[PolymarketMarket]:
        """Fetch active markets from Polymarket"""
        try:
            # Use Polymarket subgraph or API
            # This is a simplified version
            query = """
            {
                markets(
                    first: %s,
                    where: { active: true },
                    orderBy: volume,
                    orderDirection: desc
                ) {
                    id
                    question
                    description
                    endTime
                    volume
                    liquidity
                    lastPrice
                    conditionId
                    resolutionSource
                    active
                    outcomes
                }
            }
            """ % limit
            
            # In production, implement actual GraphQL query
            # For now, return mock data structure
            
            markets = []
            
            return markets
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    async def fetch_market_data(self, market_id: str) -> Optional[PolymarketMarket]:
        """Fetch detailed market data"""
        try:
            # Fetch market data from subgraph
            # This would be implemented with actual GraphQL queries
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None
    
    async def fetch_order_books(self, market_id: str) -> Dict[str, OrderBook]:
        """Fetch order books for all outcomes of a market"""
        order_books = {}
        
        try:
            # Fetch market to get outcomes
            market = await self.fetch_market_data(market_id)
            if not market:
                return {}
            
            for outcome in market.outcomes:
                # In production, fetch actual order book from API
                # This is a placeholder implementation
                
                order_books[outcome] = OrderBook(
                    outcome=outcome,
                    bids=[(Decimal('0.45'), Decimal('100')), (Decimal('0.44'), Decimal('50'))],
                    asks=[(Decimal('0.55'), Decimal('100')), (Decimal('0.56'), Decimal('50'))],
                    last_trade=(Decimal('0.52'), Decimal('10'))
                )
            
            return order_books
            
        except Exception as e:
            logger.error(f"Error fetching order books: {e}")
            return {}
    
    # ==================== STRATEGIES ====================
    
    async def arbitrage_opportunities(self) -> List[TradingSignal]:
        """
        Find arbitrage opportunities across Polymarket markets
        Inspired by: https://github.com/Polymarket/agents
        
        Opportunities:
        1. Cross-market arbitrage (same question on different markets)
        2. Price vs probability mispricing
        3. Resolution source mispricing
        """
        signals = []
        
        try:
            # Fetch all active markets
            markets = await self.fetch_active_markets(limit=100)
            
            # Group markets by similar questions
            market_groups = self._group_similar_markets(markets)
            
            for group in market_groups.values():
                if len(group) < 2:
                    continue
                
                # Find price discrepancies
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        market1 = group[i]
                        market2 = group[j]
                        
                        # Get order books
                        order_book1 = await self.fetch_order_books(market1.market_id)
                        order_book2 = await self.fetch_order_books(market2.market_id)
                        
                        # Find arbitrage opportunities
                        arb_signals = self._find_arbitrage(
                            market1, order_book1, 
                            market2, order_book2
                        )
                        
                        signals.extend(arb_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error finding arbitrage: {e}")
            return []
    
    def _group_similar_markets(self, markets: List[PolymarketMarket]) -> Dict[str, List[PolymarketMarket]]:
        """Group markets by similar questions"""
        groups = {}
        
        for market in markets:
            # Simple grouping by keywords
            key = market.question.lower()[:50]  # First 50 chars
            if key not in groups:
                groups[key] = []
            groups[key].append(market)
        
        return groups
    
    def _find_arbitrage(self, 
                       market1: PolymarketMarket, 
                       order_book1: Dict[str, OrderBook],
                       market2: PolymarketMarket,
                       order_book2: Dict[str, OrderBook]) -> List[TradingSignal]:
        """Find arbitrage between two markets"""
        signals = []
        
        # This is a simplified implementation
        # In production, you would:
        # 1. Match outcomes between markets
        # 2. Calculate implied probabilities
        # 3. Find mispricings
        # 4. Calculate profit after fees
        
        return signals
    
    async def spike_detection_strategy(self) -> List[TradingSignal]:
        """
        Detect and trade price spikes in Polymarket
        Inspired by: https://github.com/Trust412/Polymarket-spike-bot-v1
        
        Strategy:
        1. Monitor markets for unusual volume/price movements
        2. Detect spikes indicating new information
        3. Trade in direction of spike with proper risk management
        """
        signals = []
        
        try:
            markets = await self.fetch_active_markets(limit=50)
            
            for market in markets:
                # Fetch recent trades
                recent_trades = await self._fetch_recent_trades(market.market_id)
                
                if len(recent_trades) < 10:
                    continue
                
                # Detect spikes
                spike_detected = self._detect_spike(recent_trades)
                
                if spike_detected:
                    # Get current order book
                    order_books = await self.fetch_order_books(market.market_id)
                    
                    # Generate trade signal based on spike direction
                    signal = self._generate_spike_signal(market, recent_trades, order_books)
                    
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in spike detection: {e}")
            return []
    
    async def _fetch_recent_trades(self, market_id: str, limit: int = 100) -> List[Dict]:
        """Fetch recent trades for a market"""
        # Implement actual trade fetching from subgraph
        return []
    
    def _detect_spike(self, trades: List[Dict]) -> bool:
        """Detect price spikes in recent trades"""
        if len(trades) < 10:
            return False
        
        # Calculate price changes
        prices = [float(t['price']) for t in trades[-10:]]
        
        # Calculate volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Detect spike (price move > 3 standard deviations)
        if len(returns) > 5:
            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            
            latest_return = returns[-1]
            z_score = abs(latest_return - mean_return) / std_return if std_return > 0 else 0
            
            return z_score > 3
        
        return False
    
    def _generate_spike_signal(self, 
                              market: PolymarketMarket,
                              trades: List[Dict],
                              order_books: Dict[str, OrderBook]) -> Optional[TradingSignal]:
        """Generate trading signal based on spike detection"""
        
        # Determine spike direction
        recent_prices = [float(t['price']) for t in trades[-5:]]
        if len(recent_prices) < 2:
            return None
        
        price_change = recent_prices[-1] - recent_prices[0]
        
        if abs(price_change) < 0.01:  # Less than 1% change
            return None
        
        # Determine which outcome to trade
        # For binary markets, spike up = buy YES, spike down = buy NO
        
        if price_change > 0:
            outcome = "YES" if "YES" in market.outcomes else market.outcomes[0]
            position_type = PositionType.YES
            confidence = min(0.7, abs(price_change) * 10)
        else:
            outcome = "NO" if "NO" in market.outcomes else market.outcomes[-1]
            position_type = PositionType.NO
            confidence = min(0.7, abs(price_change) * 10)
        
        # Get best price from order book
        if outcome in order_books:
            order_book = order_books[outcome]
            if order_book.asks:
                best_price = float(order_book.asks[0][0])
            else:
                best_price = float(market.last_price)
        else:
            best_price = float(market.last_price)
        
        # Calculate position size using Kelly criterion
        kelly_fraction = self._calculate_kelly_fraction(market, outcome, best_price)
        amount = Decimal(str(self.max_position_size * Decimal(str(kelly_fraction * 0.5))))  # Use half-kelly
        
        return TradingSignal(
            market_id=market.market_id,
            outcome=outcome,
            position_type=position_type,
            amount=amount,
            price=Decimal(str(best_price)),
            confidence=confidence,
            strategy="spike_detection",
            kelly_fraction=kelly_fraction
        )
    
    async def copy_trading_strategy(self, 
                                  trader_addresses: List[str],
                                  min_success_rate: float = 0.6) -> List[TradingSignal]:
        """
        Copy successful traders on Polymarket
        Inspired by: https://github.com/Trust412/polymarket-copy-trading-bot-v1
        
        Strategy:
        1. Identify successful traders
        2. Analyze their trading patterns
        3. Copy their trades with delay management
        """
        signals = []
        
        try:
            for trader_address in trader_addresses:
                # Fetch trader's recent trades
                trader_trades = await self._fetch_trader_trades(trader_address)
                
                if not trader_trades:
                    continue
                
                # Calculate trader performance
                performance = self._analyze_trader_performance(trader_trades)
                
                if performance['success_rate'] < min_success_rate:
                    continue
                
                # Get trader's recent trades (last 24 hours)
                recent_trades = [t for t in trader_trades 
                               if t['timestamp'] > time.time() - 86400]
                
                for trade in recent_trades:
                    # Check if we should copy this trade
                    if self._should_copy_trade(trade, performance):
                        signal = await self._create_copy_trade_signal(trade)
                        if signal:
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in copy trading: {e}")
            return []
    
    async def _fetch_trader_trades(self, trader_address: str) -> List[Dict]:
        """Fetch all trades for a specific trader"""
        # Implement using Polymarket subgraph
        return []
    
    def _analyze_trader_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Analyze trader's historical performance"""
        if not trades:
            return {'success_rate': 0, 'avg_profit': 0, 'total_trades': 0}
        
        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
        success_rate = len(profitable_trades) / len(trades)
        
        avg_profit = sum(t.get('profit', 0) for t in trades) / len(trades) if trades else 0
        
        return {
            'success_rate': success_rate,
            'avg_profit': avg_profit,
            'total_trades': len(trades)
        }
    
    def _should_copy_trade(self, trade: Dict, performance: Dict) -> bool:
        """Determine if we should copy a specific trade"""
        
        # Basic filters
        min_trade_age = 60  # Don't copy trades older than 60 seconds
        max_position_size = Decimal('500')  # Max position size to copy
        
        trade_age = time.time() - trade['timestamp']
        if trade_age > min_trade_age:
            return False
        
        if Decimal(str(trade.get('amount', 0))) > max_position_size:
            return False
        
        # Check if trader has good performance in this market
        # Could add more sophisticated filters
        
        return True
    
    async def _create_copy_trade_signal(self, trade: Dict) -> Optional[TradingSignal]:
        """Create trading signal from copied trade"""
        try:
            market_id = trade['marketId']
            outcome = trade['outcome']
            
            # Fetch current market data
            market = await self.fetch_market_data(market_id)
            if not market:
                return None
            
            # Get current order book
            order_books = await self.fetch_order_books(market_id)
            
            if outcome not in order_books:
                return None
            
            order_book = order_books[outcome]
            
            # Use the same price as the trader if available, otherwise use best ask
            if trade.get('price'):
                price = Decimal(str(trade['price']))
            elif order_book.asks:
                price = order_book.asks[0][0]
            else:
                return None
            
            # Determine position type
            if outcome.upper() == "YES":
                position_type = PositionType.YES
            elif outcome.upper() == "NO":
                position_type = PositionType.NO
            else:
                position_type = PositionType.YES  # Default
            
            # Use smaller position size than original trader
            amount = Decimal(str(trade.get('amount', 0))) * Decimal('0.5')
            
            return TradingSignal(
                market_id=market_id,
                outcome=outcome,
                position_type=position_type,
                amount=amount,
                price=price,
                confidence=0.65,  # Moderate confidence for copy trading
                strategy="copy_trading",
                expected_profit=float(trade.get('expected_profit', 0))
            )
            
        except Exception as e:
            logger.error(f"Error creating copy trade signal: {e}")
            return None
    
    async def market_making_strategy(self, 
                                   market_id: str,
                                   spread: float = 0.02) -> List[TradingSignal]:
        """
        Market making strategy for Polymarket
        Provide liquidity on both sides of the order book
        
        Args:
            market_id: Market to provide liquidity for
            spread: Bid-ask spread as percentage
        """
        signals = []
        
        try:
            market = await self.fetch_market_data(market_id)
            if not market:
                return []
            
            order_books = await self.fetch_order_books(market_id)
            
            for outcome, order_book in order_books.items():
                if not order_book.bids or not order_book.asks:
                    continue
                
                current_bid = float(order_book.bids[0][0])
                current_ask = float(order_book.asks[0][0])
                current_mid = (current_bid + current_ask) / 2
                
                # Calculate our quotes
                our_bid = Decimal(str(current_mid * (1 - spread/2)))
                our_ask = Decimal(str(current_mid * (1 + spread/2)))
                
                # Place bids if our bid is better than current bid
                if our_bid > Decimal(str(current_bid)):
                    signals.append(TradingSignal(
                        market_id=market_id,
                        outcome=outcome,
                        position_type=PositionType.YES if "YES" in outcome else PositionType.YES,
                        amount=Decimal('50'),  # Fixed amount for market making
                        price=our_bid,
                        confidence=0.5,
                        strategy="market_making"
                    ))
                
                # Place asks if our ask is better than current ask
                if our_ask < Decimal(str(current_ask)):
                    signals.append(TradingSignal(
                        market_id=market_id,
                        outcome=outcome,
                        position_type=PositionType.NO if "NO" in outcome else PositionType.NO,
                        amount=Decimal('50'),
                        price=our_ask,
                        confidence=0.5,
                        strategy="market_making"
                    ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in market making: {e}")
            return []
    
    def _calculate_kelly_fraction(self, 
                                 market: PolymarketMarket, 
                                 outcome: str, 
                                 price: float) -> float:
        """
        Calculate Kelly criterion fraction for position sizing
        
        Kelly Criterion: f* = (bp - q) / b
        where:
        f* = fraction of capital to bet
        b = net odds received on the bet (b = (1/p) - 1)
        p = probability of winning
        q = probability of losing (q = 1 - p)
        """
        
        # Convert price to implied probability
        p = price  # In prediction markets, price ≈ probability
        
        if p <= 0 or p >= 1:
            return 0.1  # Default to 10%
        
        # Calculate b (net odds)
        b = (1 / p) - 1
        
        # Kelly fraction
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0
        
        # Use half-kelly for risk management
        half_kelly = kelly / 2
        
        # Cap between 5% and 25%
        return max(0.05, min(0.25, half_kelly))
    
    async def execute_trade(self, signal: TradingSignal) -> Dict:
        """Execute a trade on Polymarket"""
        
        if not self.account:
            return {'status': 'error', 'message': 'No trading account configured'}
        
        try:
            # Prepare trade parameters
            market = await self.fetch_market_data(signal.market_id)
            if not market:
                return {'status': 'error', 'message': 'Market not found'}
            
            # Get current gas price
            gas_price = int(self.w3.eth.gas_price * self.gas_price_multiplier)
            
            # Build transaction
            # Note: This is simplified - actual Polymarket trading is more complex
            # and involves conditional token positions
            
            tx = {
                'from': self.address,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.address),
            }
            
            # In production, you would:
            # 1. Create position in conditional tokens
            # 2. Or use Polymarket's trading interface
            # 3. Handle slippage and order execution
            
            # For now, return mock execution
            return {
                'status': 'success',
                'transaction_hash': '0x' + '0' * 64,  # Mock hash
                'signal': signal,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def monitor_positions(self) -> List[Dict]:
        """Monitor current positions and manage risk"""
        positions_status = []
        
        try:
            # Fetch all active markets we have positions in
            for market_id, position in self.positions.items():
                market = await self.fetch_market_data(market_id)
                if not market:
                    continue
                
                # Calculate current value
                current_price = market.last_price
                position_value = position['amount'] * current_price
                
                # Calculate PnL
                entry_value = position['amount'] * position['entry_price']
                pnl = position_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                # Check stop loss / take profit
                should_close = False
                close_reason = ""
                
                if position.get('stop_loss') and pnl_pct < -position['stop_loss']:
                    should_close = True
                    close_reason = "stop_loss"
                elif position.get('take_profit') and pnl_pct > position['take_profit']:
                    should_close = True
                    close_reason = "take_profit"
                
                positions_status.append({
                    'market_id': market_id,
                    'outcome': position['outcome'],
                    'amount': position['amount'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'should_close': should_close,
                    'close_reason': close_reason
                })
            
            return positions_status
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return []
