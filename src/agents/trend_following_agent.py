from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig

class TrendFollowingAgent:
    def __init__(
        self,
        config: BinanceConfig,
        data_provider: BinanceDataProvider,
        executor: BinanceExecutor
    ):
        """Initialize the trend following agent."""
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Dict] = {}

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend following indicators."""
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate ADX for trend strength
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Calculate +DM and -DM
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate +DI and -DI
        tr_14 = true_range.rolling(14).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / tr_14
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr_14
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy and sell signals based on trend following strategy.
        Returns: (buy_signal, sell_signal)
        """
        if len(df) < 50:  # Need at least 50 candles for all indicators
            return False, False

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Trend strength condition
        strong_trend = last_row['adx'] > 25
        
        # EMA crossover conditions
        ema_crossover = (prev_row['ema_short'] < prev_row['ema_medium'] and 
                        last_row['ema_short'] > last_row['ema_medium'])
        ema_crossunder = (prev_row['ema_short'] > prev_row['ema_medium'] and 
                         last_row['ema_short'] < last_row['ema_medium'])
        
        # Trend direction
        uptrend = (last_row['ema_short'] > last_row['ema_medium'] > last_row['ema_long'])
        downtrend = (last_row['ema_short'] < last_row['ema_medium'] < last_row['ema_long'])
        
        # Generate signals
        buy_signal = strong_trend and uptrend and ema_crossover
        sell_signal = strong_trend and downtrend and ema_crossunder

        return buy_signal, sell_signal

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on ATR for volatility-adjusted sizing."""
        try:
            # Get historical data for ATR calculation
            df = self.data_provider.get_historical_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                limit=50
            )
            df = self.calculate_indicators(df)
            
            # Get account balance
            balances = self.data_provider.get_account_balance()
            quote_balance = balances.get(self.config.quote_currency, 0.0)
            
            # Calculate position size based on ATR
            atr = df['atr'].iloc[-1]
            risk_per_trade = quote_balance * self.config.max_position_size
            position_size = risk_per_trade / (atr * 2)  # Use 2x ATR for stop loss
            
            # Get symbol precision
            _, quantity_precision = self.executor.get_symbol_precision(symbol)
            
            # Round to appropriate precision
            position_size = round(position_size, quantity_precision)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def execute_trade(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict:
        """Execute a trade with the given parameters."""
        try:
            order = self.executor.create_order(
                symbol=symbol,
                side=side,
                order_type=self.config.default_order_type,
                quantity=quantity,
                price=price,
                time_in_force=self.config.default_time_in_force
            )
            
            # Update positions
            if side == 'BUY':
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price or float(order['price']),
                    'order_id': order['orderId']
                }
            elif side == 'SELL':
                if symbol in self.positions:
                    del self.positions[symbol]
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            raise

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit conditions are met."""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Calculate price changes
        price_change = (current_price - entry_price) / entry_price
        
        if price_change <= -self.config.stop_loss_pct:
            return 'SELL'  # Stop loss triggered
        elif price_change >= self.config.take_profit_pct:
            return 'SELL'  # Take profit triggered
            
        return None

    def run_strategy(self, symbol: str) -> None:
        """Run the trend following strategy for a given symbol."""
        try:
            # Get historical data
            df = self.data_provider.get_historical_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                limit=100
            )
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Generate signals
            buy_signal, sell_signal = self.generate_signals(df)
            
            # Get current price
            current_price = self.data_provider.get_ticker_price(symbol)
            
            # Check stop loss and take profit
            sl_tp_signal = self.check_stop_loss_take_profit(symbol, current_price)
            
            if sl_tp_signal:
                # Execute stop loss or take profit
                if symbol in self.positions:
                    self.execute_trade(
                        symbol=symbol,
                        side=sl_tp_signal,
                        quantity=self.positions[symbol]['quantity'],
                        price=current_price
                    )
            elif buy_signal and symbol not in self.positions:
                # Calculate position size and execute buy
                quantity = self.calculate_position_size(symbol, current_price)
                if quantity > 0:
                    self.execute_trade(
                        symbol=symbol,
                        side='BUY',
                        quantity=quantity,
                        price=current_price
                    )
            elif sell_signal and symbol in self.positions:
                # Execute sell
                self.execute_trade(
                    symbol=symbol,
                    side='SELL',
                    quantity=self.positions[symbol]['quantity'],
                    price=current_price
                )
                
        except Exception as e:
            self.logger.error(f"Error running strategy for {symbol}: {e}")

    def run_all_symbols(self) -> None:
        """Run the strategy for all configured trading pairs."""
        for symbol in self.config.trading_pairs:
            self.run_strategy(symbol)

    def _calculate_ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def consult_crypto(self, symbol, timeframe, model_used="trend-following", liquidation=False):
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=120)
        latest_close = df['close'].iloc[-1]
        ema_fast = self._calculate_ema(df['close'], 20)
        ema_slow = self._calculate_ema(df['close'], 100)
        # Calculate ADX for trend strength
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr_14 = true_range.rolling(14).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / tr_14
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr_14
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        # Trend filter
        strong_trend = adx > 20
        if liquidation:
            signal = "LIQUIDATION"
            action = "HOLD"
            confidence = 0
            reasoning = "Account liquidated. No further trades."
            entry_condition = "N/A"
            exit_condition = "N/A"
        elif ema_fast.iloc[-1] > ema_slow.iloc[-1] and strong_trend:
            signal = "BULLISH"
            action = "LONG"
            confidence = 85
            reasoning = "EMA(20) above EMA(100) and ADX > 20. Strong uptrend."
            entry_condition = f"Open LONG if price stays above EMA(20) ({self.format_price(ema_fast.iloc[-1])})"
            exit_condition = f"Close LONG if price drops below EMA(100) ({self.format_price(ema_slow.iloc[-1])})"
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and strong_trend:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 85
            reasoning = "EMA(20) below EMA(100) and ADX > 20. Strong downtrend."
            entry_condition = f"Open SHORT if price stays below EMA(20) ({self.format_price(ema_fast.iloc[-1])})"
            exit_condition = f"Close SHORT if price rises above EMA(100) ({self.format_price(ema_slow.iloc[-1])})"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No strong trend or ADX too low."
            entry_condition = "Wait for clear trend and ADX > 20."
            exit_condition = "N/A"
        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {self.format_price(latest_close)}. "
            f"EMA(20): {self.format_price(ema_fast.iloc[-1])}, EMA(100): {self.format_price(ema_slow.iloc[-1])}, ADX: {adx:.2f}."
        )
        return {
            "agent": "trend_following",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Trend following position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If price crosses EMAs in opposite direction, consider stop-loss.",
            "stop_loss_pct": 0.03,  # Tighter stop
            "take_profit_pct": 0.08,  # Quicker profit taking
            "suggested_duration": "Ride the trend.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {self.format_price(latest_close)} | EMA(20): {self.format_price(ema_fast.iloc[-1])} | EMA(100): {self.format_price(ema_slow.iloc[-1])} | ADX: {adx:.2f}`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Reversal:** If price crosses EMAs in opposite direction, consider stop-loss.\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        }

    def format_price(self, price):
        if price >= 1:
            return f"${price:.2f}"
        elif price >= 0.01:
            return f"${price:.4f}"
        else:
            return f"${price:.6f}" 