from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig

class BinanceStrategy:
    def __init__(
        self,
        config: BinanceConfig,
        data_provider: BinanceDataProvider,
        executor: BinanceExecutor
    ):
        """Initialize the Binance trading strategy."""
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Dict] = {}  # Current positions

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']

        return df

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy and sell signals based on technical indicators.
        Returns: (buy_signal, sell_signal)
        """
        if len(df) < self.config.macd_slow:
            return False, False

        last_row = df.iloc[-1]
        
        # RSI conditions
        rsi_oversold = last_row['rsi'] < self.config.rsi_oversold
        rsi_overbought = last_row['rsi'] > self.config.rsi_overbought
        
        # MACD conditions
        macd_crossover = (df['macd'].iloc[-2] < df['signal'].iloc[-2] and 
                         df['macd'].iloc[-1] > df['signal'].iloc[-1])
        macd_crossunder = (df['macd'].iloc[-2] > df['signal'].iloc[-2] and 
                          df['macd'].iloc[-1] < df['signal'].iloc[-1])

        # Generate signals
        buy_signal = rsi_oversold and macd_crossover
        sell_signal = rsi_overbought and macd_crossunder

        return buy_signal, sell_signal

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate the position size based on risk management rules."""
        try:
            # Get account balance
            balances = self.data_provider.get_account_balance()
            quote_balance = balances.get(self.config.quote_currency, 0.0)
            
            # Calculate position size based on max position size
            max_position_value = quote_balance * self.config.max_position_size
            position_size = max_position_value / price
            
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
        """Run the trading strategy for a given symbol."""
        try:
            # Get historical data
            df = self.data_provider.get_historical_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                limit=100  # Get last 100 candles
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

    def consult_crypto(self, symbol, timeframe, window_df=None, model_used="binance_strategy"):
        # Use window_df if provided, else fetch data
        if window_df is not None:
            df = window_df.copy()
        else:
            df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=100)
        df = self.calculate_indicators(df)
        latest_close = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        signal_line = df['signal'].iloc[-1]
        buy_signal, sell_signal = self.generate_signals(df)
        if buy_signal:
            signal = "BULLISH"
            action = "LONG"
            confidence = 80
            reasoning = "RSI is oversold and MACD bullish crossover."
            entry_condition = f"Buy if RSI < {self.config.rsi_oversold} and MACD crosses above signal."
            exit_condition = "Sell if RSI > overbought or MACD crosses below signal."
        elif sell_signal:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 80
            reasoning = "RSI is overbought and MACD bearish crossunder."
            entry_condition = f"Sell if RSI > {self.config.rsi_overbought} and MACD crosses below signal."
            exit_condition = "Buy if RSI < oversold or MACD crosses above signal."
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No strong signal from RSI or MACD."
            entry_condition = "Wait for clear signal."
            exit_condition = "N/A"
        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {latest_close:.2f}. "
            f"RSI: {rsi:.2f}. MACD: {macd:.2f}, Signal: {signal_line:.2f}."
        )
        return {
            "agent": "binance_strategy",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "RSI/MACD-based position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If MACD/RSI reverses, consider stop-loss.",
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.10,
            "suggested_duration": "Hold until signal reverses.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {latest_close:.2f} | RSI: {rsi:.2f} | MACD: {macd:.2f} | Signal: {signal_line:.2f}`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Reversal:** If MACD/RSI reverses, consider stop-loss.\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        } 