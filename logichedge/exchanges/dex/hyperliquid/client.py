import json
import logging
from typing import Dict, List, Optional, Any
import requests
from web3 import Web3

from .types import Order, OrderSide, OrderType, Position, Balance, Ticker
from .utils import generate_signature, timestamp, handle_response, HyperliquidAPIError

class HyperliquidClient:
    """
    Hyperliquid DEX client for spot and perpetual trading.
    Based on patterns from asterdex-hl-trading-bot and Aster repos.
    """
    
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz", 
                 api_key: Optional[str] = None, 
                 private_key: Optional[str] = None,
                 testnet: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.private_key = private_key
        self.testnet = testnet
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Set appropriate base URL for testnet/mainnet
        if testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        
        # Initialize Web3 for signing if private key provided
        self.web3 = Web3() if private_key else None
        
    def _get_headers(self, requires_auth: bool = False) -> Dict[str, str]:
        """Generate request headers"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'LogicHedge/1.0'
        }
        
        if requires_auth and self.api_key:
            headers['X-API-KEY'] = self.api_key
            
        return headers
    
    def _sign_message(self, message: Dict[str, Any]) -> str:
        """Sign message with private key for DEX transactions"""
        if not self.private_key or not self.web3:
            raise HyperliquidAPIError("Private key required for signing")
        
        message_str = json.dumps(message, separators=(',', ':'))
        signed_message = self.web3.eth.account.sign_message(
            self.web3.eth.account.messages.encode_defunct(text=message_str),
            private_key=self.private_key
        )
        return signed_message.signature.hex()
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including supported symbols"""
        endpoint = f"{self.base_url}/info"
        response = self.session.get(endpoint, headers=self._get_headers())
        return handle_response(response)
    
    def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a symbol"""
        endpoint = f"{self.base_url}/ticker"
        params = {'symbol': symbol}
        response = self.session.get(endpoint, params=params, headers=self._get_headers())
        data = handle_response(response)
        
        return Ticker(
            symbol=symbol,
            bid_price=float(data.get('bidPrice', 0)),
            ask_price=float(data.get('askPrice', 0)),
            last_price=float(data.get('lastPrice', 0)),
            volume_24h=float(data.get('volume24h', 0)),
            price_change_24h=float(data.get('priceChange24h', 0))
        )
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get orderbook for a symbol"""
        endpoint = f"{self.base_url}/orderbook"
        params = {'symbol': symbol, 'depth': depth}
        response = self.session.get(endpoint, params=params, headers=self._get_headers())
        return handle_response(response)
    
    def get_balances(self) -> List[Balance]:
        """Get account balances"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for private endpoints")
            
        endpoint = f"{self.base_url}/account/balances"
        headers = self._get_headers(requires_auth=True)
        response = self.session.get(endpoint, headers=headers)
        data = handle_response(response)
        
        balances = []
        for asset_data in data.get('balances', []):
            balances.append(Balance(
                asset=asset_data['asset'],
                total=float(asset_data['total']),
                available=float(asset_data['available']),
                locked=float(asset_data['locked'])
            ))
        
        return balances
    
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for private endpoints")
            
        endpoint = f"{self.base_url}/account/positions"
        headers = self._get_headers(requires_auth=True)
        response = self.session.get(endpoint, headers=headers)
        data = handle_response(response)
        
        positions = []
        for pos_data in data.get('positions', []):
            positions.append(Position(
                symbol=pos_data['symbol'],
                size=float(pos_data['size']),
                entry_price=float(pos_data['entryPrice']),
                liquidation_price=float(pos_data.get('liquidationPrice', 0)),
                unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                realized_pnl=float(pos_data.get('realizedPnl', 0)),
                margin_used=float(pos_data.get('marginUsed', 0)),
                leverage=float(pos_data.get('leverage', 1))
            ))
        
        return positions
    
    def place_order(self, order: Order) -> Dict[str, Any]:
        """Place a new order"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for trading")
            
        endpoint = f"{self.base_url}/order"
        headers = self._get_headers(requires_auth=True)
        
        order_data = {
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': str(order.quantity),
            'timeInForce': order.time_in_force.value,
            'reduceOnly': order.reduce_only
        }
        
        if order.price is not None:
            order_data['price'] = str(order.price)
            
        if order.client_order_id:
            order_data['clientOrderId'] = order.client_order_id
        
        # Sign the order if private key is available (for DEX settlement)
        if self.private_key:
            signature = self._sign_message(order_data)
            order_data['signature'] = signature
        
        response = self.session.post(endpoint, json=order_data, headers=headers)
        return handle_response(response)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for trading")
            
        endpoint = f"{self.base_url}/order"
        headers = self._get_headers(requires_auth=True)
        
        cancel_data = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        if self.private_key:
            signature = self._sign_message(cancel_data)
            cancel_data['signature'] = signature
        
        response = self.session.delete(endpoint, json=cancel_data, headers=headers)
        return handle_response(response)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for private endpoints")
            
        endpoint = f"{self.base_url}/account/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        headers = self._get_headers(requires_auth=True)
        response = self.session.get(endpoint, params=params, headers=headers)
        data = handle_response(response)
        
        return data.get('orders', [])
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for private endpoints")
            
        endpoint = f"{self.base_url}/account/orderHistory"
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
            
        headers = self._get_headers(requires_auth=True)
        response = self.session.get(endpoint, params=params, headers=headers)
        data = handle_response(response)
        
        return data.get('orders', [])
