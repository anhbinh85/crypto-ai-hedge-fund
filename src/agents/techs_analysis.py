import os
import pandas as pd
import pandas_ta as ta
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import re
from src.data.binance_provider import BinanceDataProvider
import logging

# Indicator periods (defaults, can be parameterized)
SMA_PERIOD = 20
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH_K = 3
OBV_SMA_PERIOD = 10
ORDER_BOOK_DEPTH = 20
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20

LONGEST_PERIOD = max(SMA_PERIOD, RSI_PERIOD, VOLUME_MA_PERIOD, MACD_SLOW, ATR_PERIOD, STOCH_K, OBV_SMA_PERIOD)
KLINE_LIMIT = LONGEST_PERIOD + 55

# --- Formatting helpers ---
def format_number(value, decimals=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "Infinity"
    try:
        value_decimal = Decimal(str(value))
        if value_decimal == 0:
            if decimals == 0: return "0"
            format_string = "0." + "0" * decimals
            return format_string
        format_string = "0." + "0" * decimals
        return str(value_decimal.quantize(Decimal(format_string), rounding=ROUND_HALF_UP))
    except Exception:
        return str(value)

def format_price(price_val):
    if price_val is None or (isinstance(price_val, float) and math.isnan(price_val)):
        return "N/A"
    if isinstance(price_val, float) and math.isinf(price_val):
        return "Infinity"
    try:
        price_decimal = Decimal(str(price_val))
        if price_decimal > 1000:
            return format_number(price_val, 2)
        elif price_decimal > 1:
            return format_number(price_val, 4)
        elif price_decimal > 0:
            return format_number(price_val, 6)
        else:
            return "0.00"
    except Exception:
        return str(price_val)

def format_volume(volume_val):
    if volume_val is None or (isinstance(volume_val, float) and math.isnan(volume_val)):
        return "N/A"
    if isinstance(volume_val, float) and math.isinf(volume_val):
        return "Infinity"
    try:
        volume = Decimal(str(volume_val))
        if volume > 1_000_000_000:
            return f"{volume / 1_000_000_000:.2f}B"
        elif volume > 1_000_000:
            return f"{volume / 1_000_000:.2f}M"
        elif volume > 1_000:
            return f"{volume / 1_000:.2f}K"
        else:
            return format_number(volume_val, 2 if volume > 1 else 6)
    except Exception:
        return str(volume_val)

def get_techs_embed(ticker, candle, binance_provider: BinanceDataProvider):
    logging.info(f"get_techs_embed called with ticker={ticker}, candle={candle}")
    try:
        # Use get_symbol_ticker for current price
        ticker_data = binance_provider.client.get_symbol_ticker(symbol=ticker)
        current_price_str = ticker_data.get('price')
        # Try to get 24h change from get_ticker if available, else set to N/A
        try:
            ticker_24h = binance_provider.client.get_ticker(symbol=ticker)
            price_change_percent_str = ticker_24h.get('priceChangePercent')
        except Exception:
            price_change_percent_str = None
    except Exception as e:
        logging.exception(f"Failed to fetch ticker data for {ticker}: {e}")
        return {"error": f"Failed to fetch ticker data for {ticker}"}
    try:
        # Use get_historical_klines for kline data
        kline_data = binance_provider.client.get_historical_klines(symbol=ticker, interval=candle, limit=KLINE_LIMIT)
        # If the result is a DataFrame, convert to list of lists for compatibility
        if isinstance(kline_data, pd.DataFrame):
            kline_data = kline_data.reset_index().values.tolist()
    except Exception as e:
        logging.exception(f"Failed to fetch kline data for {ticker} {candle}: {e}")
        return {"error": f"Failed to fetch kline data for {ticker} {candle}"}
    try:
        depth_data = binance_provider.client.get_order_book(symbol=ticker, limit=ORDER_BOOK_DEPTH)
    except Exception as e:
        logging.exception(f"Failed to fetch order book for {ticker}: {e}")
        return {"error": f"Failed to fetch order book for {ticker}"}

    # 2. Prepare Kline Data
    df = pd.DataFrame(kline_data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df['Timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'Quote asset volume',
                    'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # S/R
    latest_close_price = df.iloc[-1]['close']
    lookback_period = min(SMA_PERIOD, len(df))
    recent_data = df.iloc[-lookback_period:]
    support_level = recent_data['low'].min()
    resistance_level = recent_data['high'].max()

    # 3. Calculate Indicators
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.rsi(length=RSI_PERIOD, append=True)
    df.ta.sma(close='volume', length=VOLUME_MA_PERIOD, append=True, col_names=(f'VOL_SMA_{VOLUME_MA_PERIOD}',))
    df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    df.ta.stoch(k=STOCH_K, d=STOCH_D, smooth_k=STOCH_SMOOTH_K, append=True)
    df.ta.obv(append=True)
    df.ta.sma(close='OBV', length=OBV_SMA_PERIOD, append=True, col_names=(f'OBV_SMA_{OBV_SMA_PERIOD}',))
    df.ta.atr(length=ATR_PERIOD, append=True)
    df.ta.cdl_pattern(name="all", append=True)

    latest_data = df.iloc[-1]
    latest_sma = latest_data.get(f'SMA_{SMA_PERIOD}', float('nan'))
    latest_rsi = latest_data.get(f'RSI_{RSI_PERIOD}', float('nan'))
    latest_volume = latest_data.get('volume', float('nan'))
    latest_volume_sma = latest_data.get(f'VOL_SMA_{VOLUME_MA_PERIOD}', float('nan'))
    macd_col_name = f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    macd_signal_col_name = f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    macd_hist_col_name = f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}'
    latest_macd_line = latest_data.get(macd_col_name, float('nan'))
    latest_macd_signal = latest_data.get(macd_signal_col_name, float('nan'))
    latest_macd_hist = latest_data.get(macd_hist_col_name, float('nan'))
    atr_col_name_base = f'ATR_{ATR_PERIOD}'
    atr_col_name_rma = f'ATRr_{ATR_PERIOD}'
    if atr_col_name_rma in latest_data:
        latest_atr = latest_data.get(atr_col_name_rma, float('nan'))
    elif atr_col_name_base in latest_data:
        latest_atr = latest_data.get(atr_col_name_base, float('nan'))
    else:
        latest_atr = float('nan')
    stoch_k_col_name = f'STOCHk_{STOCH_K}_{STOCH_D}_{STOCH_SMOOTH_K}'
    stoch_d_col_name = f'STOCHd_{STOCH_K}_{STOCH_D}_{STOCH_SMOOTH_K}'
    latest_stoch_k = latest_data.get(stoch_k_col_name, float('nan'))
    latest_stoch_d = latest_data.get(stoch_d_col_name, float('nan'))
    latest_obv = latest_data.get('OBV', float('nan'))
    latest_obv_sma = latest_data.get(f'OBV_SMA_{OBV_SMA_PERIOD}', float('nan'))

    # Candlestick patterns
    candlestick_patterns = []
    for col in df.columns:
        if col.startswith('CDL') and not pd.isna(latest_data.get(col)) and latest_data.get(col, 0) != 0:
            match = re.match(r"CDL([A-Z]+)", col, re.IGNORECASE)
            if match:
                pattern_name = match.group(1)
            else:
                pattern_name = col
            signal = latest_data.get(col)
            direction = "📈 Bullish" if signal > 0 else "📉 Bearish" if signal < 0 else "⚪ Neutral"
            candlestick_patterns.append(f"{pattern_name} ({direction})")

    # Order book
    bids = depth_data.get('bids', [])
    asks = depth_data.get('asks', [])
    largest_bid_price = "N/A"
    largest_bid_qty = 0
    largest_ask_price = "N/A"
    largest_ask_qty = 0
    if bids:
        max_bid_qty = 0
        max_bid_price = ""
        for price_str, qty_str in bids[:ORDER_BOOK_DEPTH]:
            try:
                qty = Decimal(qty_str)
                if qty > max_bid_qty:
                    max_bid_qty = qty
                    max_bid_price = price_str
            except Exception: continue
        if max_bid_qty > 0:
            largest_bid_price = format_price(max_bid_price)
            largest_bid_qty = max_bid_qty
    if asks:
        max_ask_qty = 0
        max_ask_price = ""
        for price_str, qty_str in asks[:ORDER_BOOK_DEPTH]:
            try:
                qty = Decimal(qty_str)
                if qty > max_ask_qty:
                    max_ask_qty = qty
                    max_ask_price = price_str
            except Exception: continue
        if max_ask_qty > 0:
            largest_ask_price = format_price(max_ask_price)
            largest_ask_qty = max_ask_qty

    # --- Formatting for Discord Embed ---
    volume_text = ""
    rsi_text = " (Neutral)"
    stoch_text = " (Neutral)"
    macd_interpretation = "Neutral"
    obv_interpretation = "Neutral"
    price_vs_sma_text = "Neutral"
    price_vs_sma_emoji = "⚪"
    macd_emoji = "⚪"
    obv_emoji = "⚪"
    overall_conclusion = "⚪ Neutral / Mixed"
    color = 0x808080
    sentiment_score = 0
    num_signals = 0

    # Price vs SMA
    if not math.isnan(latest_close_price) and not math.isnan(latest_sma):
        num_signals += 1
        if latest_close_price > latest_sma:
            price_vs_sma_text = f"Above SMA({SMA_PERIOD})"
            price_vs_sma_emoji = "🟢"
            color = 0x00FF00
            sentiment_score += 1
        elif latest_close_price < latest_sma:
            price_vs_sma_text = f"Below SMA({SMA_PERIOD})"
            price_vs_sma_emoji = "🔴"
            color = 0xFF0000
            sentiment_score -= 1
        else:
            price_vs_sma_text = f"On SMA({SMA_PERIOD})"
            price_vs_sma_emoji = "🟠"
            color = 0xFFA500

    # Volume vs SMA
    if not math.isnan(latest_volume) and not math.isnan(latest_volume_sma) and latest_volume_sma > 0:
        if latest_volume > latest_volume_sma * 1.5: volume_text = " (High)"
        elif latest_volume < latest_volume_sma * 0.7: volume_text = " (Low)"

    # RSI
    if not math.isnan(latest_rsi):
        num_signals += 1
        if latest_rsi > RSI_OVERBOUGHT:
            rsi_text = " (Overbought!)"
            sentiment_score -= 0.5
        elif latest_rsi < RSI_OVERSOLD:
            rsi_text = " (Oversold!)"
            sentiment_score += 0.5
        elif latest_rsi > 50: sentiment_score += 0.25
        elif latest_rsi < 50: sentiment_score -= 0.25
        else: rsi_text = " (Neutral)"

    # Stochastic
    if not math.isnan(latest_stoch_k):
        num_signals += 1
        if latest_stoch_k > STOCH_OVERBOUGHT:
            stoch_text = " (Overbought!)"
            sentiment_score -= 0.5
        elif latest_stoch_k < STOCH_OVERSOLD:
            stoch_text = " (Oversold!)"
            sentiment_score += 0.5
        else: stoch_text = " (Neutral)"

    # MACD
    if not math.isnan(latest_macd_line) and not math.isnan(latest_macd_signal):
        num_signals += 1
        is_bullish_cross = latest_macd_line > latest_macd_signal
        is_bearish_cross = latest_macd_line < latest_macd_signal
        hist_positive = not math.isnan(latest_macd_hist) and latest_macd_hist > 0
        hist_negative = not math.isnan(latest_macd_hist) and latest_macd_hist < 0
        if is_bullish_cross and hist_positive:
            macd_interpretation = "Bullish 🟢"
            macd_emoji = "🟢"
            sentiment_score += 1.5
        elif is_bearish_cross and hist_negative:
            macd_interpretation = "Bearish 🔴"
            macd_emoji = "🔴"
            sentiment_score -= 1.5
        elif is_bullish_cross:
            macd_interpretation = "Bullish Cross? 🟡"
            macd_emoji = "🟡"
            sentiment_score += 0.5
        elif is_bearish_cross:
            macd_interpretation = "Bearish Cross? 🟡"
            macd_emoji = "🟡"
            sentiment_score -= 0.5
        else: macd_interpretation = "Neutral"; macd_emoji = "⚪"

    # OBV
    if not math.isnan(latest_obv) and not math.isnan(latest_obv_sma):
        num_signals += 1
        if latest_obv > latest_obv_sma:
            obv_interpretation = "Rising (Buy Pressure)"
            obv_emoji = "🟢"
            sentiment_score += 1
        elif latest_obv < latest_obv_sma:
            obv_interpretation = "Falling (Sell Pressure)"
            obv_emoji = "🔴"
            sentiment_score -= 1
        else: obv_interpretation = "Neutral"; obv_emoji = "⚪"

    # Candlestick Pattern Sentiment
    bullish_patterns_count = sum(1 for p in candlestick_patterns if "Bullish" in p)
    bearish_patterns_count = sum(1 for p in candlestick_patterns if "Bearish" in p)
    if bullish_patterns_count > 0 or bearish_patterns_count > 0:
        num_signals += 1
        sentiment_score += 0.75 * bullish_patterns_count
        sentiment_score -= 0.75 * bearish_patterns_count

    # Overall conclusion
    if num_signals > 0:
        normalized_score = sentiment_score / num_signals
        if normalized_score >= 0.7: overall_conclusion = "🟢 Strong Bullish"
        elif normalized_score >= 0.3: overall_conclusion = "🟡 Mildly Bullish"
        elif normalized_score <= -0.7: overall_conclusion = "🔴 Strong Bearish"
        elif normalized_score <= -0.3: overall_conclusion = "🟠 Mildly Bearish"
        else: overall_conclusion = "⚪ Neutral / Mixed"
        if rsi_text == " (Oversold!)" and stoch_text == " (Oversold!)":
            overall_conclusion += " (Potential Oversold Bounce?)"
        elif rsi_text == " (Overbought!)" and stoch_text == " (Overbought!)":
            overall_conclusion += " (Potential Overbought Pullback?)"

    formatted_price = format_price(latest_close_price)
    formatted_sma = format_price(latest_sma)
    formatted_rsi_val = format_number(latest_rsi, 2)
    formatted_change = f"{Decimal(str(price_change_percent_str)):+.2f}%" if price_change_percent_str is not None else "N/A"
    formatted_volume = format_volume(latest_volume)
    formatted_volume_sma = format_volume(latest_volume_sma)
    formatted_support = format_price(support_level)
    formatted_resistance = format_price(resistance_level)
    formatted_largest_bid_qty = format_volume(largest_bid_qty) if largest_bid_qty > 0 else "N/A"
    formatted_largest_ask_qty = format_volume(largest_ask_qty) if largest_ask_qty > 0 else "N/A"
    formatted_atr = format_price(latest_atr)
    formatted_stoch_k = format_number(latest_stoch_k, 2)
    formatted_stoch_d = format_number(latest_stoch_d, 2)
    formatted_macd_line = format_number(latest_macd_line, 4)
    formatted_macd_signal = format_number(latest_macd_signal, 4)
    formatted_macd_hist = format_number(latest_macd_hist, 4)
    formatted_obv = format_volume(latest_obv)
    formatted_obv_sma = format_volume(latest_obv_sma)
    atr_percentage = "N/A"
    if not math.isnan(latest_atr) and not math.isnan(latest_close_price) and latest_close_price > 0:
        atr_percentage = f"{format_number((latest_atr / latest_close_price) * 100, 2)}%"
    candlestick_text = ", ".join(candlestick_patterns) if candlestick_patterns else "None Detected"
    if len(candlestick_text) > 1000:
        candlestick_text = candlestick_text[:1000] + "..."
    now_utc = datetime.now(timezone.utc)
    formatted_timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S.') + f"{now_utc.microsecond // 1000:03d}Z"
    embed = {
        "title": f"📊 {ticker} Analysis ({candle})",
        "description": f"**Overall Sentiment: {overall_conclusion}**",
        "color": color,
        "fields": [
            {"name": "Price", "value": f"${formatted_price}", "inline": True},
            {"name": "24h Change", "value": str(formatted_change), "inline": True},
            {"name": f"Price vs SMA({SMA_PERIOD})", "value": f"{price_vs_sma_emoji} {price_vs_sma_text}", "inline": True},
            {"name": f"Support ({SMA_PERIOD}p Low)", "value": f"${formatted_support}", "inline": True},
            {"name": f"Resistance ({SMA_PERIOD}p High)", "value": f"${formatted_resistance}", "inline": True},
            {"name": "Volatility (ATR)", "value": f"${formatted_atr} ({atr_percentage})", "inline": True},
            {"name": f"RSI({RSI_PERIOD})", "value": f"{formatted_rsi_val}{rsi_text}", "inline": True},
            {"name": f"Stoch({STOCH_K},{STOCH_D}) %K", "value": f"{formatted_stoch_k}{stoch_text}", "inline": True},
            {"name": "Stoch %D", "value": str(formatted_stoch_d), "inline": True},
            {"name": f"MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})", "value": f"**{macd_interpretation}**", "inline": False},
            {"name": "MACD Line", "value": str(formatted_macd_line), "inline": True},
            {"name": "Signal Line", "value": str(formatted_macd_signal), "inline": True},
            {"name": "Histogram", "value": str(formatted_macd_hist), "inline": True},
            {"name": f"Volume{volume_text}", "value": str(formatted_volume), "inline": True},
            {"name": f"Vol SMA({VOLUME_MA_PERIOD})", "value": str(formatted_volume_sma), "inline": True},
            {"name": f"OBV ({obv_emoji})", "value": f"{obv_interpretation}", "inline": True},
            {"name": f"OBV Value", "value": str(formatted_obv), "inline": True},
            {"name": f"OBV SMA({OBV_SMA_PERIOD})", "value": str(formatted_obv_sma), "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "🕯️ Candlestick Patterns (Last Candle)", "value": candlestick_text, "inline": False},
            {"name": f"Largest Buy Wall (Top {ORDER_BOOK_DEPTH})", "value": f"{formatted_largest_bid_qty} @ ${largest_bid_price}", "inline": True},
            {"name": f"Largest Sell Wall (Top {ORDER_BOOK_DEPTH})", "value": f"{formatted_largest_ask_qty} @ ${largest_ask_price}", "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
        ],
        "timestamp": formatted_timestamp,
        "footer": {"text": f"Data: Binance | Bot v2.0"}
    }
    return {"embeds": [embed]} 