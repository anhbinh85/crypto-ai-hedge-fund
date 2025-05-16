from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig

class MeanReversionAgent:
    def __init__(
        self,
        config: BinanceConfig,
        data_provider: BinanceDataProvider,
        executor: BinanceExecutor
    ):
        """Initialize the mean reversion agent."""
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Dict] = {}

    def _calculate_bollinger(self, series, window=20):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return sma, upper, lower

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.isnull().all() else 0

    def consult_crypto(self, symbol, timeframe, model_used="mean-reversion"):
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=40)
        latest_close = df['close'].iloc[-1]
        sma, upper, lower = self._calculate_bollinger(df['close'])
        rsi = self._calculate_rsi(df['close'])

        if latest_close < lower.iloc[-1] and rsi < 30:
            signal = "BULLISH"
            action = "LONG"
            confidence = 80
            reasoning = "Price below lower Bollinger Band and oversold RSI. Mean reversion likely."
            entry_condition = f"Open LONG if price rises above ${latest_close * 1.02:.2f}"
            exit_condition = f"Close LONG if price returns to SMA (${sma.iloc[-1]:.2f})"
        elif latest_close > upper.iloc[-1] and rsi > 70:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 80
            reasoning = "Price above upper Bollinger Band and overbought RSI. Mean reversion likely."
            entry_condition = f"Open SHORT if price drops below ${latest_close * 0.98:.2f}"
            exit_condition = f"Close SHORT if price returns to SMA (${sma.iloc[-1]:.2f})"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No mean reversion signal."
            entry_condition = "Wait for extreme Bollinger Band touch."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: ${latest_close:.2f}. "
            f"SMA(20): ${sma.iloc[-1]:.2f}, Upper: ${upper.iloc[-1]:.2f}, Lower: ${lower.iloc[-1]:.2f}, RSI(14): {rsi:.2f}."
        )

        return {
            "agent": "mean_reversion",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Mean reversion position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If price moves 5% against position, consider stop-loss.",
            "suggested_duration": "Until price returns to mean.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: ${latest_close:.2f} | SMA(20): ${sma.iloc[-1]:.2f} | Upper: ${upper.iloc[-1]:.2f} | Lower: ${lower.iloc[-1]:.2f} | RSI(14): {rsi:.2f}`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Reversal:** If price moves 5% against position, consider stop-loss.\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Generate buy and sell signals based on mean reversion strategy.
        Returns: (buy_signal, sell_signal)
        """
        if len(df) < 20:  # Need at least 20 candles for indicators
            return False, False

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # Mean reversion conditions
        price_below_lower = last_row['close'] < last_row['lower_band']
        price_above_upper = last_row['close'] > last_row['upper_band']
        
        # RSI conditions
        rsi_oversold = last_row['rsi'] < 30
        rsi_overbought = last_row['rsi'] > 70
        
        # Stochastic conditions
        stoch_oversold = last_row['stoch_k'] < 20 and last_row['stoch_d'] < 20
        stoch_overbought = last_row['stoch_k'] > 80 and last_row['stoch_d'] > 80
        
        # Generate signals
        buy_signal = (price_below_lower and rsi_oversold) or stoch_oversold
        sell_signal = (price_above_upper and rsi_overbought) or stoch_overbought

        return buy_signal, sell_signal

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on volatility."""
        try:
            # Get historical data for volatility calculation
            df = self.data_provider.get_historical_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                limit=20
            )
            df = self.calculate_indicators(df)
            
            # Get account balance
            balances = self.data_provider.get_account_balance()
            quote_balance = balances.get(self.config.quote_currency, 0.0)
            
            # Calculate position size based on volatility
            volatility = df['std'].iloc[-1] / df['sma'].iloc[-1]  # Coefficient of variation
            risk_adjusted_size = self.config.max_position_size * (1 - volatility)
            position_size = (quote_balance * risk_adjusted_size) / price
            
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
        """Run the mean reversion strategy for a given symbol."""
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