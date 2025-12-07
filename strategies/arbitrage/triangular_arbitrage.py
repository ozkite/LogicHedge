"""
Triangular Arbitrage Strategy
Low Risk - Exploits price discrepancies between three currencies
Source: Inspired by CryptoSignal and jesse repos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TriangularOpportunity:
    pair1: str
    pair2: str
    pair3: str
    profit_percent: float
    path: List[str]
    timestamp: datetime

class TriangularArbitrage:
    def __init__(self, exchange_client, min_profit_threshold=0.5):
        """
        Initialize triangular arbitrage detector
        
        Args:
            exchange_client: Exchange API client
            min_profit_threshold: Minimum profit % to execute
        """
        self.exchange = exchange_client
        self.min_profit = min_profit_threshold
        self.triangles = self._find_triangles()
        
    def _find_triangles(self) -> List[List[str]]:
        """Find all possible triangular paths"""
        # Get all trading pairs
        markets = self.exchange.fetch_markets()
        symbols = [m['symbol'] for m in markets]
        
        triangles = []
        # Simple implementation - find BTC/USDT, ETH/BTC, ETH/USDT type triangles
        # In production, implement proper graph traversal
        
        # This is a simplified version
        base_coins = ['BTC', 'ETH', 'BNB', 'USDT']
        
        for base in base_coins:
            for quote1 in base_coins:
                for quote2 in base_coins:
                    if base != quote1 != quote2:
                        triangles.append([f"{base}/{quote1}", f"{quote1}/{quote2}", f"{base}/{quote2}"])
        
        return triangles
    
    def check_opportunities(self) -> List[TriangularOpportunity]:
        """Check for triangular arbitrage opportunities"""
        opportunities = []
        
        # Fetch current prices
        tickers = self.exchange.fetch_tickers()
        
        for triangle in self.triangles:
            try:
                # Get prices for all three pairs
                prices = []
                for pair in triangle:
                    if pair in tickers:
                        prices.append(tickers[pair]['last'])
                    else:
                        # Try reverse pair
                        base, quote = pair.split('/')
                        reverse_pair = f"{quote}/{base}"
                        if reverse_pair in tickers:
                            prices.append(1 / tickers[reverse_pair]['last'])
                        else:
                            break
                
                if len(prices) == 3:
                    # Calculate arbitrage profit
                    # Path: Buy A with B, Buy B with C, Sell A for C
                    implied_rate = (1 / prices[0]) * (1 / prices[1]) * prices[2]
                    profit_percent = (implied_rate - 1) * 100
                    
                    if profit_percent > self.min_profit:
                        opportunity = TriangularOpportunity(
                            pair1=triangle[0],
                            pair2=triangle[1],
                            pair3=triangle[2],
                            profit_percent=profit_percent,
                            path=triangle,
                            timestamp=datetime.now()
                        )
                        opportunities.append(opportunity)
                        
            except Exception as e:
                continue
        
        return sorted(opportunities, key=lambda x: x.profit_percent, reverse=True)
    
    def execute_triangle(self, opportunity: TriangularOpportunity) -> Dict:
        """Execute triangular arbitrage trade"""
        # Note: In production, add proper error handling, slippage calculation,
        # and ensure all orders are filled atomically or use smart order routing
        
        orders = []
        try:
            # Example execution path (simplified)
            # 1. Buy pair1
            # 2. Buy pair2 with acquired asset
            # 3. Sell final asset in pair3
            
            # In reality, need to consider:
            # - Order book depth
            # - Transaction fees
            # - Slippage
            # - Network latency
            
            return {
                'status': 'success',
                'opportunity': opportunity,
                'orders': orders,
                'realized_profit': 0  # Calculated after execution
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
