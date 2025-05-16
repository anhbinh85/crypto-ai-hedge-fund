from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import pandas_ta as ta

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
        self.logger.setLevel(logging.INFO)  # Ensure INFO logs are shown
        self.positions: Dict[str, Dict] = {}  # Current positions

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        if df is None:
            raise ValueError("Input DataFrame is None. No historical data available for indicator calculation.")
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

        # SMA(20)
        df['sma20'] = df['close'].rolling(window=20).mean()

        # Ichimoku
        ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku_df is not None:
            for col in ichimoku_df.columns:
                df[col] = ichimoku_df[col]

        # ADX(14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            for col in adx.columns:
                df[col] = adx[col]

        # Stochastic(14,3)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            for col in stoch.columns:
                df[col] = stoch[col]

        # Stoch RSI(14)
        stochrsi = ta.stochrsi(df['close'], length=14)
        if stochrsi is not None:
            for col in stochrsi.columns:
                df[col] = stochrsi[col]

        # ATR(14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Volume SMA(20)
        df['vol_sma20'] = df['volume'].rolling(window=20).mean()

        # VWAP
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['vwap'] = vwap

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
        import math
        if window_df is not None:
            df = window_df.copy()
        else:
            df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=120)
        # Add detailed logging for debugging
        self.logger.info(f"Fetched klines for {symbol} {timeframe}: type={type(df)}, is None={df is None}")
        print(f"[DEBUG] Fetched klines for {symbol} {timeframe}: type={type(df)}, is None={df is None}")
        if df is not None:
            self.logger.info(f"DataFrame shape: {df.shape}, columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")
            print(f"[DEBUG] DataFrame shape: {df.shape}, columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")
            try:
                self.logger.info(f"DataFrame head:\n{df.head()}\n")
                print(f"[DEBUG] DataFrame head:\n{df.head()}\n")
            except Exception as e:
                self.logger.warning(f"Could not print DataFrame head: {e}")
                print(f"[DEBUG] Could not print DataFrame head: {e}")
        if df is None or df.empty:
            self.logger.error(f"No historical data found for {symbol} on {timeframe} in consult_crypto.")
            print(f"[DEBUG] No historical data found for {symbol} on {timeframe} in consult_crypto.")
            return {
                "agent": "binance_strategy",
                "signal": "N/A",
                "confidence": 0,
                "reasoning": f"No historical data found for {symbol} on {timeframe}. Please check the symbol and timeframe.",
                "action": "HOLD",
                "quantity": 0,
                "quantity_explanation": "No data available.",
                "entry_condition": "N/A",
                "exit_condition": "N/A",
                "reversal_signal": "N/A",
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "suggested_duration": "N/A",
                "model_used": model_used,
                "data_summary": f"No data for {symbol} on {timeframe}.",
                "discord_message": f"**{symbol} ({timeframe})**\nNo historical data found. Please check the symbol and timeframe."
            }
        # Always slice to last 30 rows for indicator calculation
        if len(df) > 30:
            df = df.tail(30).copy()
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing columns {missing_cols} for {symbol} on {timeframe} in consult_crypto. Columns present: {list(df.columns)}")
            print(f"[DEBUG] Missing columns {missing_cols} for {symbol} on {timeframe} in consult_crypto. Columns present: {list(df.columns)}")
            return {
                "agent": "binance_strategy",
                "signal": "N/A",
                "confidence": 0,
                "reasoning": f"Missing required data columns: {', '.join(missing_cols)} for {symbol} on {timeframe}. Please check the symbol and timeframe.",
                "action": "HOLD",
                "quantity": 0,
                "quantity_explanation": "No data available.",
                "entry_condition": "N/A",
                "exit_condition": "N/A",
                "reversal_signal": "N/A",
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "suggested_duration": "N/A",
                "model_used": model_used,
                "data_summary": f"No data for {symbol} on {timeframe}.",
                "discord_message": f"**{symbol} ({timeframe})**\nMissing required data columns: {', '.join(missing_cols)}. Please check the symbol and timeframe."
            }
        # Check for minimum data length
        if len(df) < 30:
            self.logger.warning(f"Not enough data for {symbol} on {timeframe}: only {len(df)} rows. Minimum required: 30.")
            print(f"[DEBUG] Not enough data for {symbol} on {timeframe}: only {len(df)} rows. Minimum required: 30.")
            return {
                "agent": "binance_strategy",
                "signal": "N/A",
                "confidence": 0,
                "reasoning": f"Not enough data for {symbol} on {timeframe}. Got {len(df)} rows, need at least 30 for analysis.",
                "action": "HOLD",
                "quantity": 0,
                "quantity_explanation": "Not enough data.",
                "entry_condition": "N/A",
                "exit_condition": "N/A",
                "reversal_signal": "N/A",
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "suggested_duration": "N/A",
                "model_used": model_used,
                "data_summary": f"Not enough data for {symbol} on {timeframe}.",
                "discord_message": f"**{symbol} ({timeframe})**\nNot enough data for analysis. Please try a different timeframe or reduce observations."
            }
        # Robust error handling for indicator calculation
        try:
            df = self.calculate_indicators(df)
        except Exception as e:
            self.logger.error(f"Indicator calculation failed for {symbol} on {timeframe}: {e}")
            print(f"[DEBUG] Indicator calculation failed for {symbol} on {timeframe}: {e}")
            return {
                "agent": "binance_strategy",
                "signal": "N/A",
                "confidence": 0,
                "reasoning": f"Indicator calculation failed: {e}",
                "action": "HOLD",
                "quantity": 0,
                "quantity_explanation": "Indicator calculation failed.",
                "entry_condition": "N/A",
                "exit_condition": "N/A",
                "reversal_signal": "N/A",
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "suggested_duration": "N/A",
                "model_used": model_used,
                "data_summary": f"Indicator calculation failed for {symbol} on {timeframe}.",
                "discord_message": f"**{symbol} ({timeframe})**\nIndicator calculation failed: {e}"
            }
        # Check for NaN in key indicators
        latest = df.iloc[-1]
        key_indicators = ['sma20', 'rsi', 'macd', 'signal', 'ADX_14', 'STOCHk_14_3_3', 'STOCHRSIk_14_14_3_3']
        for ind in key_indicators:
            if ind not in latest or pd.isna(latest[ind]):
                self.logger.warning(f"Key indicator {ind} is NaN or missing for {symbol} on {timeframe}.")
                print(f"[DEBUG] Key indicator {ind} is NaN or missing for {symbol} on {timeframe}.")
                return {
                    "agent": "binance_strategy",
                    "signal": "N/A",
                    "confidence": 0,
                    "reasoning": f"Key indicator {ind} is NaN or missing. Not enough data for analysis.",
                    "action": "HOLD",
                    "quantity": 0,
                    "quantity_explanation": "Key indicator missing or NaN.",
                    "entry_condition": "N/A",
                    "exit_condition": "N/A",
                    "reversal_signal": "N/A",
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "suggested_duration": "N/A",
                    "model_used": model_used,
                    "data_summary": f"Key indicator {ind} missing or NaN for {symbol} on {timeframe}.",
                    "discord_message": f"**{symbol} ({timeframe})**\nKey indicator {ind} missing or NaN. Not enough data for analysis."
                }
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        # Extract indicators
        close = latest['close']
        sma20 = latest.get('sma20', float('nan'))
        rsi = latest.get('rsi', float('nan'))
        macd = latest.get('macd', float('nan'))
        signal_line = latest.get('signal', float('nan'))
        adx = latest.get('ADX_14', float('nan'))
        stoch_k = latest.get('STOCHk_14_3_3', float('nan'))
        stoch_d = latest.get('STOCHd_14_3_3', float('nan'))
        stochrsi_k = latest.get('STOCHRSIk_14_14_3_3', float('nan'))
        stochrsi_d = latest.get('STOCHRSId_14_14_3_3', float('nan'))
        atr = latest.get('atr', float('nan'))
        vol = latest.get('volume', float('nan'))
        vol_sma20 = latest.get('vol_sma20', float('nan'))
        vwap = latest.get('vwap', float('nan'))
        ichimoku_conv = latest.get('ITS_9', float('nan'))
        ichimoku_base = latest.get('IKS_26', float('nan'))
        ichimoku_a = latest.get('ISA_9', float('nan'))
        ichimoku_b = latest.get('ISB_26', float('nan'))
        # Signal logic
        macd_bullish = prev['macd'] < prev['signal'] and macd > signal_line
        macd_bearish = prev['macd'] > prev['signal'] and macd < signal_line
        adx_strong = adx > 20 if not math.isnan(adx) else False
        stochrsi_oversold = stochrsi_k < 0.2 if not math.isnan(stochrsi_k) else False
        stochrsi_overbought = stochrsi_k > 0.8 if not math.isnan(stochrsi_k) else False
        price_above_sma = close > sma20 if not math.isnan(sma20) else False
        price_below_sma = close < sma20 if not math.isnan(sma20) else False
        # Decision
        if (
            rsi < self.config.rsi_oversold and
            macd_bullish and
            adx_strong and
            stochrsi_oversold and
            price_above_sma
        ):
            signal = "BULLISH"
            action = "LONG"
            confidence = 90
            reasoning = "RSI oversold, MACD bullish crossover, ADX strong, Stoch RSI oversold, price above SMA(20)."
            entry_condition = f"Buy if all bullish conditions met."
            exit_condition = "Sell if any bullish condition fails."
        elif (
            rsi > self.config.rsi_overbought and
            macd_bearish and
            adx_strong and
            stochrsi_overbought and
            price_below_sma
        ):
            signal = "BEARISH"
            action = "SHORT"
            confidence = 90
            reasoning = "RSI overbought, MACD bearish crossunder, ADX strong, Stoch RSI overbought, price below SMA(20)."
            entry_condition = f"Sell if all bearish conditions met."
            exit_condition = "Buy if any bearish condition fails."
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No strong multi-indicator signal."
            entry_condition = "Wait for clear multi-indicator signal."
            exit_condition = "N/A"
        # Format numbers for summary
        def fmt(val, dec=2):
            if val is None or (isinstance(val, float) and (math.isnan(val) or val == float('nan'))):
                return "N/A"
            return f"{val:.{dec}f}"
        data_summary = (
            f"{symbol} on {timeframe} timeframe.\n"
            f"Price: {fmt(close)} | SMA(20): {fmt(sma20)} | RSI: {fmt(rsi)} | MACD: {fmt(macd)} | Signal: {fmt(signal_line)}\n"
            f"ADX(14): {fmt(adx)} | Stoch(14,3): {fmt(stoch_k)}/{fmt(stoch_d)} | Stoch RSI: {fmt(stochrsi_k,2)}/{fmt(stochrsi_d,2)}\n"
            f"ATR(14): {fmt(atr)} | Vol: {fmt(vol)} | Vol SMA(20): {fmt(vol_sma20)} | VWAP: {fmt(vwap)}\n"
            f"Ichimoku Conv/Base/A/B: {fmt(ichimoku_conv)}/{fmt(ichimoku_base)}/{fmt(ichimoku_a)}/{fmt(ichimoku_b)}"
        )
        return {
            "agent": "binance_strategy",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Multi-indicator position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If multi-indicator signal reverses, consider stop-loss.",
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.10,
            "suggested_duration": "Hold until signal reverses.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {fmt(close)} | SMA(20): {fmt(sma20)} | RSI: {fmt(rsi)} | MACD: {fmt(macd)} | Signal: {fmt(signal_line)}`\n"
                f"`ADX(14): {fmt(adx)} | Stoch(14,3): {fmt(stoch_k)}/{fmt(stoch_d)} | Stoch RSI: {fmt(stochrsi_k,2)}/{fmt(stochrsi_d,2)}`\n"
                f"`ATR(14): {fmt(atr)} | Vol: {fmt(vol)} | Vol SMA(20): {fmt(vol_sma20)} | VWAP: {fmt(vwap)}`\n"
                f"`Ichimoku Conv/Base/A/B: {fmt(ichimoku_conv)}/{fmt(ichimoku_base)}/{fmt(ichimoku_a)}/{fmt(ichimoku_b)}`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Reversal:** If multi-indicator signal reverses, consider stop-loss.\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        } 