from typing import Dict, List, Optional, Tuple
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import logging
from ratelimit import limits, RateLimitException
import time

# Global rate limiter: 5700 requests per minute
BINANCE_REQUESTS_PER_MIN = 5700
BINANCE_PERIOD = 60  # seconds

def binance_rate_limiter():
    # Dummy function for decorator
    pass

@limits(calls=BINANCE_REQUESTS_PER_MIN, period=BINANCE_PERIOD)
def _rate_limited_binance_call():
    pass

# Decorator to use for all Binance API calls
def rate_limited(func):
    def wrapper(*args, **kwargs):
        _rate_limited_binance_call()
        return func(*args, **kwargs)
    return wrapper

class BinanceDataProvider:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize Binance data provider with API credentials."""
        self.client = Client(api_key, api_secret)
        self.logger = logging.getLogger(__name__)

    @rate_limited
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical klines (candlestick data) from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to milliseconds timestamp
            start_ts = int(start_time.timestamp() * 1000) if start_time else None
            end_ts = int(end_time.timestamp() * 1000) if end_time else None
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise

    @rate_limited
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance for all assets."""
        try:
            account = self.client.get_account()
            balances = {}
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free > 0 or locked > 0:
                    balances[asset['asset']] = free + locked
            return balances
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching account balance: {e}")
            raise

    @rate_limited
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading pair information."""
        try:
            return self.client.get_symbol_info(symbol)
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching symbol info: {e}")
            raise

    @rate_limited
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching ticker price: {e}")
            raise

    @rate_limited
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol."""
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching order book: {e}")
            raise

    @rate_limited
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent trades for a symbol."""
        try:
            return self.client.get_recent_trades(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching recent trades: {e}")
            raise 