from typing import Dict, Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from decimal import Decimal, ROUND_DOWN
from src.data.binance_provider import rate_limited

class BinanceExecutor:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize Binance executor with API credentials."""
        self.client = Client(api_key, api_secret)
        self.logger = logging.getLogger(__name__)

    @rate_limited
    def get_symbol_precision(self, symbol: str) -> Tuple[int, int]:
        """Get price and quantity precision for a symbol."""
        try:
            info = self.client.get_symbol_info(symbol)
            price_precision = 0
            quantity_precision = 0
            
            for filter in info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    price_precision = len(str(float(filter['tickSize'])).rstrip('0').split('.')[-1])
                elif filter['filterType'] == 'LOT_SIZE':
                    quantity_precision = len(str(float(filter['stepSize'])).rstrip('0').split('.')[-1])
            
            return price_precision, quantity_precision
        except BinanceAPIException as e:
            self.logger.error(f"Error getting symbol precision: {e}")
            raise

    @rate_limited
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = 'GTC'
    ) -> Dict:
        """
        Create a new order on Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', etc.
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: 'GTC', 'IOC', 'FOK'
            
        Returns:
            Order response from Binance
        """
        try:
            # Get symbol precision
            price_precision, quantity_precision = self.get_symbol_precision(symbol)
            
            # Round quantity and price to appropriate precision
            quantity = Decimal(str(quantity)).quantize(
                Decimal('0.' + '0' * quantity_precision),
                rounding=ROUND_DOWN
            )
            
            if price:
                price = Decimal(str(price)).quantize(
                    Decimal('0.' + '0' * price_precision),
                    rounding=ROUND_DOWN
                )
            
            # Create order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': float(quantity),
                'timeInForce': time_in_force
            }
            
            if price:
                params['price'] = float(price)
            
            return self.client.create_order(**params)
            
        except BinanceAPIException as e:
            self.logger.error(f"Error creating order: {e}")
            raise

    @rate_limited
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an existing order."""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error(f"Error canceling order: {e}")
            raise

    @rate_limited
    def get_order(self, symbol: str, order_id: int) -> Dict:
        """Get order status."""
        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error(f"Error getting order: {e}")
            raise

    @rate_limited
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get all open orders for a symbol or all symbols."""
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            return self.client.get_open_orders()
        except BinanceAPIException as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise

    @rate_limited
    def get_all_orders(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> list:
        """Get all orders for a symbol."""
        try:
            return self.client.get_all_orders(
                symbol=symbol,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
        except BinanceAPIException as e:
            self.logger.error(f"Error getting all orders: {e}")
            raise 