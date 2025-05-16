import os
import pandas as pd
import pandas_ta as ta
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import re
from src.data.binance_provider import BinanceDataProvider
import logging
from src.data.cache import get_cache
import time

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
KLINE_LIMIT = 200

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
    cache = get_cache()
    cache_key = f"{ticker}_{candle}"
    now = time.time()
    # Try to get cached klines (with timestamp)
    kline_cache = cache.get_prices(cache_key)
    if kline_cache and isinstance(kline_cache, dict):
        kline_data = kline_cache.get('data')
        ts = kline_cache.get('ts', 0)
        if kline_data and (now - ts) < 60:
            use_cache = True
        else:
            use_cache = False
    else:
        use_cache = False
    if not use_cache:
        try:
            kline_data = binance_provider.client.get_historical_klines(symbol=ticker, interval=candle, limit=KLINE_LIMIT)
            # Save to cache with timestamp
            cache.set_prices(cache_key, {'data': kline_data, 'ts': now})
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
    # --- Additional Indicators ---
    # 1. Bollinger Bands
    bb = df.ta.bbands(append=True)
    # 2. Ichimoku Cloud
    ichimoku = df.ta.ichimoku(append=True)
    # 3. ADX
    adx = df.ta.adx(append=True)
    # 4. Keltner Channels
    kc = df.ta.kc(append=True)
    # 5. Stochastic RSI
    stochrsi = df.ta.stochrsi(append=True)
    # 6. Donchian Channels
    donchian = df.ta.donchian(append=True)
    # 7. Supertrend
    supertrend = df.ta.supertrend(append=True)
    # 8. VWAP
    vwap = df.ta.vwap(append=True)
    # 9. Chaikin Money Flow
    cmf = df.ta.cmf(append=True)
    # 10. Fisher Transform
    fisher = df.ta.fisher(append=True)
    # 11. Connors RSI
    # crsi = df.ta.crsi(append=True)  # Not available in pandas_ta, so skip
    # 12. Hull Moving Average
    hma = df.ta.hma(append=True)
    # 13. Ease of Movement
    eom = df.ta.eom(append=True)
    # 14. Klinger Oscillator
    kvo = df.ta.kvo(append=True)

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

    # --- Additional Indicator Values ---
    # Bollinger Bands
    bb_upper = latest_data.get('BBU_20_2.0', float('nan'))
    bb_middle = latest_data.get('BBM_20_2.0', float('nan'))
    bb_lower = latest_data.get('BBL_20_2.0', float('nan'))
    # Ichimoku Cloud (Span A/B, Conversion/Base)
    ichimoku_a = latest_data.get('ISA_9', float('nan'))
    ichimoku_b = latest_data.get('ISB_26', float('nan'))
    ichimoku_conv = latest_data.get('ITS_9', float('nan'))
    ichimoku_base = latest_data.get('IKS_26', float('nan'))
    # ADX
    adx_val = latest_data.get('ADX_14', float('nan'))
    # Keltner Channels
    kc_upper = latest_data.get('KCU_20_2.0_10', float('nan'))
    kc_middle = latest_data.get('KCM_20_2.0_10', float('nan'))
    kc_lower = latest_data.get('KCL_20_2.0_10', float('nan'))
    # Stochastic RSI
    stochrsi_k = latest_data.get('STOCHRSIk_14_14_3_3', float('nan'))
    stochrsi_d = latest_data.get('STOCHRSId_14_14_3_3', float('nan'))
    # Donchian Channels
    donchian_upper = latest_data.get('DCHigh_20', float('nan'))
    donchian_lower = latest_data.get('DCLow_20', float('nan'))
    # Supertrend
    supertrend_val = latest_data.get('SUPERT_10_3.0', float('nan'))
    supertrend_dir = latest_data.get('SUPERTd_10_3.0', float('nan'))
    # VWAP
    vwap_val = latest_data.get('VWAP_D', float('nan'))
    # Chaikin Money Flow
    cmf_val = latest_data.get('CMF_20', float('nan'))
    # Fisher Transform
    fisher_val = latest_data.get('FISHERT_9', float('nan'))
    # Connors RSI
    # crsi_val = latest_data.get('CRSI_3_2_100', float('nan'))  # Not available
    # Hull Moving Average
    hma_val = latest_data.get('HMA_20', float('nan'))
    # Ease of Movement
    eom_val = latest_data.get('EOM_14_100000000', float('nan'))
    # Klinger Oscillator
    kvo_val = latest_data.get('KVO_34_55_13', float('nan'))

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
    try:
        # Use get_symbol_ticker for current price
        ticker_data = binance_provider.client.get_symbol_ticker(symbol=ticker)
        current_price_str = ticker_data.get('price')
        # Always fetch 24h price change percent
        try:
            ticker_24h = binance_provider.client.get_ticker(symbol=ticker)
            price_change_percent_str = ticker_24h.get('priceChangePercent')
        except Exception:
            price_change_percent_str = None
    except Exception as e:
        logging.exception(f"Failed to fetch ticker data for {ticker}: {e}")
        return {"error": f"Failed to fetch ticker data for {ticker}"}
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

    # --- Grouped Indicator Strings (hide N/A) ---
    def hide_na(label, value, emoji=None):
        if value is None or (isinstance(value, str) and (value == "N/A" or value == "nan")):
            return None
        if isinstance(value, float) and (math.isnan(value) or value == float('nan')):
            return None
        if emoji:
            return f"{emoji} {label}: {value}"
        return f"{label}: {value}"

    # Volatility
    volatility_lines = []
    bb_val = f"{format_price(bb_upper)} / {format_price(bb_middle)} / {format_price(bb_lower)}"
    if not ("N/A" in bb_val):
        volatility_lines.append(f"📉 BB: {bb_val}")
    kc_val = f"{format_price(kc_upper)} / {format_price(kc_middle)} / {format_price(kc_lower)}"
    if not ("N/A" in kc_val):
        volatility_lines.append(f"📏 KC: {kc_val}")
    donchian_val = f"{format_price(donchian_upper)} / {format_price(donchian_lower)}"
    if not ("N/A" in donchian_val):
        volatility_lines.append(f"📊 Donchian: {donchian_val}")
    if not math.isnan(latest_atr):
        volatility_lines.append(f"📈 ATR: {formatted_atr} ({atr_percentage})")
    if not (supertrend_val == "N/A" or str(supertrend_dir) == "nan"):
        volatility_lines.append(f"🔥 Supertrend: {format_price(supertrend_val)} (Dir: {supertrend_dir})")
    volatility_text = "\n".join(volatility_lines) if volatility_lines else "-"

    # Trend
    trend_lines = []
    if not math.isnan(latest_sma):
        trend_lines.append(f"🟢 SMA({SMA_PERIOD}): {formatted_sma}")
    ichimoku_val = f"{format_price(ichimoku_conv)} / {format_price(ichimoku_base)} / {format_price(ichimoku_a)} / {format_price(ichimoku_b)}"
    if not ("N/A" in ichimoku_val):
        trend_lines.append(f"🌥️ Ichimoku: {ichimoku_val}")
    if not math.isnan(hma_val):
        trend_lines.append(f"💹 Hull MA: {format_price(hma_val)}")
    if not math.isnan(adx_val):
        trend_lines.append(f"📏 ADX(14): {format_number(adx_val, 2)}")
    trend_text = "\n".join(trend_lines) if trend_lines else "-"

    # Momentum
    momentum_lines = []
    if not math.isnan(latest_rsi):
        momentum_lines.append(f"🟦 RSI({RSI_PERIOD}): {formatted_rsi_val}{rsi_text}")
    if not math.isnan(latest_stoch_k):
        momentum_lines.append(f"🟪 Stoch({STOCH_K},{STOCH_D}) %K: {formatted_stoch_k}{stoch_text}")
    if not math.isnan(latest_stoch_d):
        momentum_lines.append(f"Stoch %D: {formatted_stoch_d}")
    stochrsi_val = f"{format_number(stochrsi_k, 2)} / {format_number(stochrsi_d, 2)}"
    if not ("N/A" in stochrsi_val):
        momentum_lines.append(f"🟫 Stoch RSI (K/D): {stochrsi_val}")
    macd_val = f"{formatted_macd_line} / {formatted_macd_signal} / {formatted_macd_hist}"
    if not ("N/A" in macd_val):
        momentum_lines.append(f"🟧 MACD: {macd_val}")
    if not math.isnan(fisher_val):
        momentum_lines.append(f"🟨 Fisher Transform: {format_number(fisher_val, 4)}")
    momentum_text = "\n".join(momentum_lines) if momentum_lines else "-"

    # Volume & Flow
    volume_lines = []
    if not math.isnan(latest_volume):
        volume_lines.append(f"🔵 Volume: {formatted_volume}{volume_text}")
    if not math.isnan(latest_volume_sma):
        volume_lines.append(f"Vol SMA({VOLUME_MA_PERIOD}): {formatted_volume_sma}")
    if not (obv_interpretation == "N/A"):
        volume_lines.append(f"🟢 OBV: {obv_interpretation} ({obv_emoji})")
    if not math.isnan(latest_obv):
        volume_lines.append(f"OBV Value: {formatted_obv}")
    if not math.isnan(latest_obv_sma):
        volume_lines.append(f"OBV SMA({OBV_SMA_PERIOD}): {formatted_obv_sma}")
    if not (vwap_val == "N/A"):
        volume_lines.append(f"💲 VWAP: {format_price(vwap_val)}")
    if not math.isnan(cmf_val):
        volume_lines.append(f"💰 Chaikin Money Flow: {format_number(cmf_val, 4)}")
    if not math.isnan(eom_val):
        volume_lines.append(f"🟠 Ease of Movement: {format_number(eom_val, 4)}")
    if not math.isnan(kvo_val):
        volume_lines.append(f"🟤 Klinger Oscillator: {format_number(kvo_val, 4)}")
    volume_text_group = "\n".join(volume_lines) if volume_lines else "-"

    # Support/Resistance
    support_resistance_lines = []
    if not (formatted_support == "N/A"):
        support_resistance_lines.append(f"🟩 Support ({SMA_PERIOD}p Low): ${formatted_support}")
    if not (formatted_resistance == "N/A"):
        support_resistance_lines.append(f"🟥 Resistance ({SMA_PERIOD}p High): ${formatted_resistance}")
    support_resistance_text = "\n".join(support_resistance_lines) if support_resistance_lines else "-"

    # Order Book
    orderbook_lines = []
    if not (formatted_largest_bid_qty == "N/A" or largest_bid_price == "N/A"):
        orderbook_lines.append(f"🟦 Largest Buy Wall: {formatted_largest_bid_qty} @ ${largest_bid_price}")
    if not (formatted_largest_ask_qty == "N/A" or largest_ask_price == "N/A"):
        orderbook_lines.append(f"🟥 Largest Sell Wall: {formatted_largest_ask_qty} @ ${largest_ask_price}")
    orderbook_text = "\n".join(orderbook_lines) if orderbook_lines else "-"

    # Candlestick Patterns (keep as is, but add emoji)
    candle_emoji = "🕯️"
    candlestick_title = f"{candle_emoji} Candlestick Patterns (Last Candle)"

    # --- Embed ---
    embed = {
        "title": f"📊 {ticker} Analysis ({candle})",
        "description": f"**Overall Sentiment:** 🟡 {overall_conclusion}",
        "color": 0xFFD700 if "Bullish" in overall_conclusion else 0xFF0000 if "Bearish" in overall_conclusion else 0x808080,
        "fields": [
            {"name": "💵 Price & Change", "value": f"${formatted_price} | 24h: {formatted_change}", "inline": True},
            {"name": "🌈 Trend Indicators", "value": trend_text, "inline": False},
            {"name": "⚡ Momentum Indicators", "value": momentum_text, "inline": False},
            {"name": "🌊 Volatility Indicators", "value": volatility_text, "inline": False},
            {"name": "🔊 Volume & Flow", "value": volume_text_group, "inline": False},
            {"name": "🛡️ Support & Resistance", "value": support_resistance_text, "inline": True},
            {"name": "📚 Order Book", "value": orderbook_text, "inline": True},
            {"name": candlestick_title, "value": candlestick_text, "inline": False},
        ],
        "timestamp": formatted_timestamp,
        "footer": {"text": f"Data: Binance | Bot v2.0"}
    }
    return {"embeds": [embed]} 