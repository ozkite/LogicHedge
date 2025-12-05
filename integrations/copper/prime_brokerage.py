"""
Copper.co institutional prime brokerage integration
Multi-sig custody with trading execution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import hmac
import hashlib
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class CopperAccount:
    """Copper institutional account"""
    account_id: str
    name: str
    type: str  # trading, custody, omnibus
    subaccounts: List[str]
    settlement_networks: List[str]
    
class CopperPrimeBrokerage:
    """
    Copper.co integration for institutional trading
    Features: Multi-sig custody, sub-accounts, OTC, settlement
    """
    
    def __init__(self, api_key: str, api_secret: str, client_id: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client_id = client_id
        self.base_url = "https://api.copper.co"
        
        # Account structure
        self.main_account = None
        self.trading_accounts = []
        self.custody_wallets = []
        
    async def initialize(self):
        """Initialize Copper connection"""
        # Get accounts
        accounts = await self._copper_request("GET", "/institutional/v1/accounts")
        
        self.main_account = CopperAccount(
            account_id=accounts['main_account']['id'],
            name=accounts['main_account']['name'],
            type=accounts['main_account']['type'],
            subaccounts=[acc['id'] for acc in accounts['subaccounts']],
            settlement_networks=accounts['settlement_networks']
        )
        
        logger.info(f"Copper initialized: {self.main_account.name}")
        
    async def create_omnibus_structure(self):
        """
        Create omnibus account structure for fund segregation
        Main account → Trading sub-accounts → Investor sub-accounts
        """
        # Create trading desk accounts
        strategies = ['arbitrage', 'market_making', 'options', 'yield']
        
        for strategy in strategies:
            account = await self._copper_request(
                "POST", "/institutional/v1/accounts",
                {
                    "name": f"trading_{strategy}",
                    "type": "trading",
                    "parent_account_id": self.main_account.account_id,
                    "permissions": ["trade", "withdraw", "deposit"]
                }
            )
            
            self.trading_accounts.append(account['id'])
            logger.info(f"Created trading account: {account['id']}")
            
    async def execute_otc_trade(self, trade_request: Dict) -> Dict:
        """
        Execute OTC trade with best execution
        """
        # Request OTC quote
        quote = await self._copper_request(
            "POST", "/otc/v1/quotes",
            {
                "side": trade_request['side'],
                "quantity": trade_request['quantity'],
                "asset": trade_request['asset'],
                "settlement_asset": trade_request.get('settlement_asset', 'USDC'),
                "account_id": trade_request['account_id']
            }
        )
        
        # Execute if price is acceptable
        if self._is_price_acceptable(quote, trade_request):
            execution = await self._copper_request(
                "POST", "/otc/v1/execute",
                {
                    "quote_id": quote['quote_id'],
                    "account_id": trade_request['account_id']
                }
            )
            
            logger.info(f"OTC trade executed: {execution['trade_id']}, "
                      f"{trade_request['quantity']} {trade_request['asset']}")
                      
            return execution
                      
    async def multi_sig_withdrawal(self, withdrawal_request: Dict) -> Dict:
        """
        Initiate multi-sig withdrawal requiring multiple approvals
        """
        # Create withdrawal request
        withdrawal = await self._copper_request(
            "POST", "/custody/v1/withdrawals",
            {
                "account_id": withdrawal_request['account_id'],
                "asset": withdrawal_request['asset'],
                "amount": withdrawal_request['amount'],
                "destination": withdrawal_request['destination'],
                "required_approvals": withdrawal_request.get('required_approvals', 2)
            }
        )
        
        # This would trigger email/SMS approvals
        logger.info(f"Multi-sig withdrawal initiated: {withdrawal['withdrawal_id']}")
        
        return withdrawal
        
    async def settle_fund_transfer(self, transfer_request: Dict) -> Dict:
        """
        Internal fund settlement between sub-accounts
        Zero on-chain transaction needed
        """
        settlement = await self._copper_request(
            "POST", "/settlement/v1/transfers",
            {
                "from_account_id": transfer_request['from_account'],
                "to_account_id": transfer_request['to_account'],
                "asset": transfer_request['asset'],
                "amount": transfer_request['amount'],
                "reference": transfer_request.get('reference', 'internal_settlement')
            }
        )
        
        logger.info(f"Internal settlement: {settlement['amount']} {settlement['asset']} "
                  f"from {settlement['from_account']} to {settlement['to_account']}")
                  
        return settlement
        
    async def get_institutional_pricing(self, asset: str, side: str, size: float) -> Dict:
        """
        Get institutional pricing (better than retail)
        """
        pricing = await self._copper_request(
            "POST", "/pricing/v1/quote",
            {
                "asset": asset,
                "side": side,
                "quantity": size,
                "tier": "institutional"  # Better pricing tier
            }
        )
        
        return {
            'price': pricing['price'],
            'fee_rate': pricing['fee_rate'],  # Lower fees
            'min_fill': pricing['min_fill'],
            'estimated_slippage': pricing['estimated_slippage']
        }
        
    async def _copper_request(self, method: str, endpoint: str, data: Dict = None):
        """Make authenticated request to Copper API"""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time()))
        nonce = str(int(time.time() * 1000))
        
        # Prepare signature
        body = json.dumps(data) if data else ''
        message = f"{timestamp}{nonce}{method}{endpoint}{body}"
        
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-CPPR-API-KEY": self.api_key,
            "X-CPPR-TIMESTAMP": timestamp,
            "X-CPPR-NONCE": nonce,
            "X-CPPR-SIGNATURE": signature,
            "X-CPPR-CLIENT": self.client_id,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, headers=headers, json=data) as response:
                    return await response.json()
