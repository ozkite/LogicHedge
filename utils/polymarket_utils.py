"""
Utility functions for Polymarket trading
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional
from decimal import Decimal
import json
import logging

logger = logging.getLogger(__name__)

class PolymarketAPI:
    """Polymarket API client"""
    
    def __init__(self, graphql_endpoint: str = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket"):
        self.endpoint = graphql_endpoint
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Execute GraphQL query"""
        if not self.session:
            async with aiohttp.ClientSession() as session:
                return await self._execute_query(session, query, variables)
        else:
            return await self._execute_query(self.session, query, variables)
    
    async def _execute_query(self, session: aiohttp.ClientSession, query: str, variables: Dict) -> Optional[Dict]:
        """Execute GraphQL query"""
        try:
            payload = {'query': query}
            if variables:
                payload['variables'] = variables
            
            async with session.post(self.endpoint, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data')
                else:
                    logger.error(f"GraphQL query failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error executing GraphQL query: {e}")
            return None
    
    async def fetch_markets(self, first: int = 50, skip: int = 0) -> List[Dict]:
        """Fetch markets from Polymarket"""
        query = """
        query GetMarkets($first: Int!, $skip: Int!) {
            markets(
                first: $first,
                skip: $skip,
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
                totalSupply
            }
        }
        """
        
        variables = {'first': first, 'skip': skip}
        
        data = await self.query(query, variables)
        if data and 'markets' in data:
            return data['markets']
        
        return []
    
    async def fetch_market_trades(self, market_id: str, first: int = 100) -> List[Dict]:
        """Fetch recent trades for a market"""
        query = """
        query GetMarketTrades($marketId: String!, $first: Int!) {
            trades(
                first: $first,
                where: { market: $marketId },
                orderBy: timestamp,
                orderDirection: desc
            ) {
                id
                trader
                outcome
                amount
                price
                timestamp
                transactionHash
            }
        }
        """
        
        variables = {'marketId': market_id, 'first': first}
        
        data = await self.query(query, variables)
        if data and 'trades' in data:
            return data['trades']
        
        return []
    
    async def fetch_trader_trades(self, trader_address: str, first: int = 100) -> List[Dict]:
        """Fetch trades by a specific trader"""
        query = """
        query GetTraderTrades($trader: String!, $first: Int!) {
            trades(
                first: $first,
                where: { trader: $trader },
                orderBy: timestamp,
                orderDirection: desc
            ) {
                id
                market {
                    id
                    question
                }
                outcome
                amount
                price
                timestamp
                transactionHash
            }
        }
        """
        
        variables = {'trader': trader_address.lower(), 'first': first}
        
        data = await self.query(query, variables)
        if data and 'trades' in data:
            return data['trades']
        
        return []

class ProbabilityCalculator:
    """Calculate and analyze probabilities for prediction markets"""
    
    @staticmethod
    def calculate_implied_probability(price: Decimal) -> Decimal:
        """Convert price to implied probability"""
        return price  # In binary markets, price = probability
    
    @staticmethod
    def calculate_expected_value(probability: Decimal, payout: Decimal, stake: Decimal) -> Decimal:
        """Calculate expected value of a bet"""
        return probability * payout - (Decimal(1) - probability) * stake
    
    @staticmethod
    def calculate_kelly_fraction(probability: Decimal, odds: Decimal) -> Decimal:
        """Calculate Kelly criterion fraction"""
        # odds = payout / stake
        q = Decimal(1) - probability
        b = odds - Decimal(1)
        
        if b <= 0:
            return Decimal(0)
        
        kelly = (b * probability - q) / b
        return max(Decimal(0), min(Decimal(1), kelly))
    
    @staticmethod
    def calculate_arbitrage_opportunity(prices: List[Decimal]) -> Optional[Dict]:
        """Find arbitrage opportunities in a set of prices"""
        if len(prices) < 2:
            return None
        
        # For binary markets, sum of YES and NO should equal 1
        total_probability = sum(prices)
        
        if total_probability < 0.98:  # Less than 1 (allowing for fees)
            # Arbitrage opportunity: buy all outcomes
            investment = Decimal(100)  # Example
            payout = investment / total_probability
            profit = payout - investment
            
            return {
                'arbitrage': True,
                'total_probability': float(total_probability),
                'expected_profit_percent': float((profit / investment) * 100),
                'stakes': [float(investment * p / total_probability) for p in prices]
            }
        
        return None
