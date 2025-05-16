from typing import Dict, List
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class BinanceConfig:
    # API Configuration
    api_key: str
    api_secret: str
    
    # Trading Parameters
    trading_pairs: List[str] = None  # List of trading pairs to monitor
    base_currency: str = 'USDT'  # Base currency for trading pairs
    quote_currency: str = 'USDT'  # Quote currency for trading pairs
    
    # Order Parameters
    default_order_type: str = 'LIMIT'  # Default order type (LIMIT, MARKET)
    default_time_in_force: str = 'GTC'  # Default time in force (GTC, IOC, FOK)
    
    # Risk Management
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    stop_loss_pct: float = 0.02  # Stop loss percentage
    take_profit_pct: float = 0.04  # Take profit percentage
    
    # Time Intervals
    candle_interval: str = '1h'  # Default candle interval
    update_interval: timedelta = timedelta(minutes=5)  # How often to update positions
    
    # Technical Analysis Parameters
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Backtesting Parameters
    backtest_start_date: str = None
    backtest_end_date: str = None
    initial_balance: float = 10000.0
    
    def __post_init__(self):
        """Initialize default trading pairs if not provided."""
        if self.trading_pairs is None:
            self.trading_pairs = [
                f'BTC{self.base_currency}',
                f'ETH{self.base_currency}',
                f'BNB{self.base_currency}',
                f'SOL{self.base_currency}',
                f'ADA{self.base_currency}'
            ]

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BinanceConfig':
        """Create a BinanceConfig instance from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert the config to a dictionary."""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'trading_pairs': self.trading_pairs,
            'base_currency': self.base_currency,
            'quote_currency': self.quote_currency,
            'default_order_type': self.default_order_type,
            'default_time_in_force': self.default_time_in_force,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'candle_interval': self.candle_interval,
            'update_interval': self.update_interval.total_seconds(),
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'backtest_start_date': self.backtest_start_date,
            'backtest_end_date': self.backtest_end_date,
            'initial_balance': self.initial_balance
        } 