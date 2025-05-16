import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import os
from src.data.binance_provider import BinanceDataProvider
from src.config.binance_config import BinanceConfig
from src.tools.binance_executor import BinanceExecutor
from src.agents.ben_graham import BenGrahamCryptoAgent
from src.agents.bill_ackman import BillAckmanCryptoAgent
from src.agents.binance_strategy import BinanceStrategy
from src.agents.mean_reversion_agent import MeanReversionAgent
from src.agents.michael_burry import MichaelBurryCryptoAgent
from src.agents.sentiment import SentimentAgent
from src.agents.trend_following_agent import TrendFollowingAgent
from src.agents.warren_buffett import WarrenBuffettCryptoAgent
from src.agents.cathie_wood import CathieWoodCryptoAgent
from src.agents.charlie_munger import CharlieMungerCryptoAgent
from src.agents.peter_lynch import PeterLynchCryptoAgent
from src.agents.stanley_druckenmiller import StanleyDruckenmillerCryptoAgent
from src.agents.phil_fisher import PhilFisherCryptoAgent
from src.agents.valuation import ValuationAgent
from src.agents.fundamentals import FundamentalsAgent
from src.agents.technicals import TechnicalsAgent
import logging

AGENT_REGISTRY = {
    "ben_graham": BenGrahamCryptoAgent,
    "bill_ackman": BillAckmanCryptoAgent,
    "binance_strategy": BinanceStrategy,
    "mean_reversion": MeanReversionAgent,
    "michael_burry": MichaelBurryCryptoAgent,
    "sentiment": SentimentAgent,
    "trend_following": TrendFollowingAgent,
    "warren_buffett": WarrenBuffettCryptoAgent,
    "cathie_wood": CathieWoodCryptoAgent,
    "charlie_munger": CharlieMungerCryptoAgent,
    "peter_lynch": PeterLynchCryptoAgent,
    "stanley_druckenmiller": StanleyDruckenmillerCryptoAgent,
    "phil_fisher": PhilFisherCryptoAgent,
    "valuation": ValuationAgent,
    "fundamentals": FundamentalsAgent,
    "technicals": TechnicalsAgent,
}

def parse_duration(duration_str):
    # e.g. '30d', '90d', '1y'
    if duration_str.endswith('d'):
        days = int(duration_str[:-1])
        return timedelta(days=days)
    elif duration_str.endswith('y'):
        years = int(duration_str[:-1])
        return timedelta(days=365*years)
    else:
        return timedelta(days=30)

def run_backtest(
    ticker="BTCUSDT",
    timeframe="1h",
    agents="all",
    observations=100,
    leverage=1,
    startup_capital=1000.0,
    export_csv=True,
    trading_fee_pct=0.0004,  # 0.04% per trade
    slippage_pct=0.0002,     # 0.02% slippage
    depth_slippage_factor=0.0001,  # extra slippage per trade size/volume
    **kwargs
):
    logger = logging.getLogger("backtest_engine")
    # Standardize ticker and timeframe
    orig_ticker = ticker
    orig_timeframe = timeframe
    ticker = ticker.upper()
    timeframe = timeframe.lower()
    # Auto-correct common user mistakes (e.g., 15M -> 15m)
    if 'M' in orig_timeframe:
        logger.info(f"Corrected timeframe from {orig_timeframe} to {timeframe} (user likely meant minutes)")
    logger.info(f"Backtest called with: ticker={ticker}, timeframe={timeframe}, agents={agents}, observations={observations}, leverage={leverage}")
    try:
        # 1. Setup
        config = BinanceConfig(api_key="", api_secret="")
        data_provider = BinanceDataProvider(config.api_key, config.api_secret)
        executor = BinanceExecutor(config.api_key, config.api_secret)
        agent_names = list(AGENT_REGISTRY.keys()) if agents == "all" else agents
        if isinstance(agent_names, str):
            agent_names = [agent_names]
        agent_classes = [AGENT_REGISTRY[name] for name in agent_names]
        agents_list = [cls(config, data_provider, executor) for cls in agent_classes]

        # 2. Get historical data (fetch exactly 'observations' candles, chunked if needed)
        max_candles = 1000
        tf_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000}
        if timeframe not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe} (original: {orig_timeframe})")
            raise Exception(f"Invalid timeframe: {timeframe}")
        seconds_per_candle = tf_map[timeframe]
        total_candles = observations
        if total_candles > 100_000:
            raise Exception(f"Requested number of candles too large for Binance API. Try a smaller number.")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=seconds_per_candle * total_candles)
        dfs = []
        chunk_start = start_time
        candles_remaining = total_candles
        while candles_remaining > 0:
            chunk_size = min(max_candles, candles_remaining)
            chunk_end = chunk_start + timedelta(seconds=seconds_per_candle * chunk_size)
            df_chunk = data_provider.get_historical_klines(
                symbol=ticker,
                interval=timeframe,
                start_time=chunk_start,
                end_time=chunk_end,
                limit=1000  # Always fetch the maximum available candles per request
            )
            if df_chunk is not None and not df_chunk.empty:
                dfs.append(df_chunk)
            chunk_start = chunk_end
            candles_remaining -= chunk_size
        # Improved error handling for empty/None DataFrames
        if not dfs or all([d is None or d.empty for d in dfs]):
            raise Exception("No historical data found for backtest.")
        df = pd.concat([d for d in dfs if d is not None and not d.empty]).drop_duplicates().sort_index()
        available_candles = len(df)
        logger.info(f"Fetched {available_candles} candles from Binance for {ticker} {timeframe}")
        if available_candles < total_candles:
            logger.error(f"Not enough historical data: requested {total_candles}, got {available_candles} for {ticker} {timeframe}")
            # If agent is binance_strategy, indicate the max possible
            if (isinstance(agents, str) and agents == "binance_strategy") or (isinstance(agents, list) and agents == ["binance_strategy"]):
                raise Exception(
                    f"Not enough historical data: requested {total_candles}, got {available_candles} for {ticker} {timeframe}. "
                    f"The maximum available for this symbol/timeframe is {available_candles-1} observations for backtesting with binance_strategy."
                )
            else:
                raise Exception(
                    f"Not enough historical data: requested {total_candles}, got {available_candles} for {ticker} {timeframe}."
                )
        if len(df) > total_candles:
            df = df.iloc[-total_candles:]
        if df.empty:
            raise Exception("No historical data found for backtest.")

        # 3. Backtest loop (ensemble agent decides action at each step)
        balance = startup_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        win_trades = 0
        loss_trades = 0
        best_trade = None
        worst_trade = None
        max_drawdown = 0
        peak = balance
        actions_log = []
        consult_logs = []  # Track consult outputs
        observation_log = []
        last_action = None
        last_position_size = 1.0

        for i in range(1, len(df)):
            window_df = df.iloc[:i+1]
            row = window_df.iloc[-1]
            price = row['close']
            # Diagnostic: Check if window_df ever has < 30 rows after warm-up
            if i >= 30 and len(window_df) < 30:
                logger.error(f"At index {i}, window_df has only {len(window_df)} rows (should be at least 30). This may indicate a data issue.")
            # --- Multi-agent ensemble ---
            signals = []
            if len(window_df) < 30:
                logger.warning(f"Skipping agent consult at {row.name}: only {len(window_df)} rows (need at least 30). Returning HOLD.")
                print(f"[DEBUG] Skipping agent consult at {row.name}: only {len(window_df)} rows (need at least 30). Returning HOLD.")
                signals = [{
                    "agent": "ensemble",
                    "signal": "N/A",
                    "confidence": 0,
                    "reasoning": f"Not enough data for analysis at {row.name}. Got {len(window_df)} rows, need at least 30.",
                    "action": "HOLD",
                    "quantity": 0,
                    "quantity_explanation": "Not enough data.",
                    "entry_condition": "N/A",
                    "exit_condition": "N/A",
                    "reversal_signal": "N/A",
                    "stop_loss_pct": 0.0,
                    "take_profit_pct": 0.0,
                    "suggested_duration": "N/A",
                    "model_used": "ensemble",
                    "data_summary": f"Not enough data for analysis at {row.name}.",
                    "discord_message": f"Not enough data for analysis at {row.name}."
                } for _ in agents_list]
            else:
                for agent in agents_list:
                    print(f"[DEBUG] Calling {agent.__class__.__name__} with window_df of length {len(window_df)} at {row.name}")
                    try:
                        signal = agent.consult_crypto(ticker, timeframe, window_df=window_df)
                    except TypeError:
                        signal = agent.consult_crypto(ticker, timeframe)
                    signals.append(signal)
            consult_logs.append({'date': row.name, 'consult_output': signals})
            # Majority vote for action
            actions = [s.get('action', 'HOLD').upper() for s in signals]
            action = max(set(actions), key=actions.count)
            # Average stop-loss/take-profit/position_size/confidence
            stop_loss_pct = np.mean([s.get('stop_loss_pct', 0.05) for s in signals])
            take_profit_pct = np.mean([s.get('take_profit_pct', 0.10) for s in signals])
            position_size = np.mean([s.get('position_size', 1.0) for s in signals])
            position_size = min(max(position_size, 0.01), 1.0)  # Clamp between 1% and 100%
            actions_log.append({'date': row.name, 'action': action})
            realized_pnl = 0
            unrealized_pnl = 0
            trade_qty = 0
            notes = ''
            stop_triggered = False
            tp_triggered = False
            reversal_triggered = False
            # --- Calculate dynamic slippage based on order book depth ---
            avg_volume = window_df['volume'].iloc[-20:].mean() * price if 'volume' in window_df.columns else 1e6
            # --- ENFORCE STOP-LOSS/TAKE-PROFIT ---
            if position != 0:
                move = (price - entry_price) / entry_price if position > 0 else (entry_price - price) / entry_price
                if move <= -stop_loss_pct:
                    # Close position at stop-loss price (with slippage)
                    close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
                    realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
                    realized_pnl *= leverage
                    # Deduct trading fee
                    fee = abs(position) * close_price * trading_fee_pct
                    balance += realized_pnl - fee
                    trade = {'date': row.name, 'action': 'STOP_LOSS', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
                    trades.append(trade)
                    notes = f'Closed {"LONG" if position > 0 else "SHORT"} (stop-loss)'
                    if realized_pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    if best_trade is None or realized_pnl > best_trade:
                        best_trade = realized_pnl
                    if worst_trade is None or realized_pnl < worst_trade:
                        worst_trade = realized_pnl
                    trade_qty = position
                    position = 0
                    entry_price = 0
                    stop_triggered = True
                elif move >= take_profit_pct:
                    # Close position at take-profit price (with slippage)
                    close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
                    realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
                    realized_pnl *= leverage
                    fee = abs(position) * close_price * trading_fee_pct
                    balance += realized_pnl - fee
                    trade = {'date': row.name, 'action': 'TAKE_PROFIT', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
                    trades.append(trade)
                    notes = f'Closed {"LONG" if position > 0 else "SHORT"} (take-profit)'
                    if realized_pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    if best_trade is None or realized_pnl > best_trade:
                        best_trade = realized_pnl
                    if worst_trade is None or realized_pnl < worst_trade:
                        worst_trade = realized_pnl
                    trade_qty = position
                    position = 0
                    entry_price = 0
                    tp_triggered = True
            # --- REVERSAL LOGIC ---
            if position != 0 and not stop_triggered and not tp_triggered:
                if (position > 0 and action in ['SHORT', 'SELL']) or (position < 0 and action in ['LONG', 'BUY']):
                    # Close current position (with slippage)
                    close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
                    realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
                    realized_pnl *= leverage
                    fee = abs(position) * close_price * trading_fee_pct
                    balance += realized_pnl - fee
                    trade = {'date': row.name, 'action': 'REVERSE', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
                    trades.append(trade)
                    notes = f'Reversed from {"LONG" if position > 0 else "SHORT"} to {action}'
                    if realized_pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    if best_trade is None or realized_pnl > best_trade:
                        best_trade = realized_pnl
                    if worst_trade is None or realized_pnl < worst_trade:
                        worst_trade = realized_pnl
                    trade_qty = position
                    position = 0
                    entry_price = 0
                    reversal_triggered = True
            # --- END ENFORCE ---
            # Only open new position if not in position and not just closed by stop-loss/take-profit/reversal
            if not stop_triggered and not tp_triggered and (position == 0):
                if action in ['LONG', 'BUY']:
                    # Open LONG with slippage and partial sizing
                    open_price = price * (1 + slippage_pct + depth_slippage_factor * (position_size * balance * leverage / price / avg_volume))
                    position = (balance * leverage * position_size) / open_price
                    entry_price = open_price
                    trade_qty = position
                    fee = abs(position) * open_price * trading_fee_pct
                    balance -= fee
                    trade = {'date': row.name, 'action': 'BUY', 'price': open_price, 'quantity': position, 'pnl': 0, 'balance': balance}
                    trades.append(trade)
                    notes = f'Opened LONG (size={position_size:.2f})'
                    last_position_size = position_size
                elif action in ['SHORT', 'SELL']:
                    # Open SHORT with slippage and partial sizing
                    open_price = price * (1 - slippage_pct - depth_slippage_factor * (position_size * balance * leverage / price / avg_volume))
                    position = -(balance * leverage * position_size) / open_price
                    entry_price = open_price
                    trade_qty = position
                    fee = abs(position) * open_price * trading_fee_pct
                    balance -= fee
                    trade = {'date': row.name, 'action': 'SELL', 'price': open_price, 'quantity': position, 'pnl': 0, 'balance': balance}
                    trades.append(trade)
                    notes = f'Opened SHORT (size={position_size:.2f})'
                    last_position_size = position_size
                elif action == 'HOLD' and last_action not in ['HOLD', None]:
                    # Close position if agent says HOLD (shouldn't happen here, but for completeness)
                    close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
                    realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
                    realized_pnl *= leverage
                    fee = abs(position) * close_price * trading_fee_pct
                    balance += realized_pnl - fee
                    trade = {'date': row.name, 'action': 'CLOSE', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
                    trades.append(trade)
                    notes = f'Closed {"LONG" if position > 0 else "SHORT"} (HOLD signal)'
                    if realized_pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    if best_trade is None or realized_pnl > best_trade:
                        best_trade = realized_pnl
                    if worst_trade is None or realized_pnl < worst_trade:
                        worst_trade = realized_pnl
                    trade_qty = position
                    position = 0
                    entry_price = 0
            last_action = action
            # Track equity
            equity = balance if position == 0 else balance + (price - entry_price) * position * leverage
            equity_curve.append(equity)
            # Drawdown
            if equity_curve[-1] > peak:
                peak = equity_curve[-1]
            dd = (peak - equity_curve[-1]) / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd
            # Calculate unrealized pnl if in position
            if position != 0:
                unrealized_pnl = (price - entry_price) * position * leverage if position > 0 else (entry_price - price) * abs(position) * leverage
            # Log observation (expanded)
            observation_log.append({
                'date': row.name,
                'ticker': ticker,
                'action': action,
                'quantity': trade_qty,
                'price': price,
                'cash': balance if position == 0 else balance - (price * abs(position)),
                'stock': position,
                'total_value': equity,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'notes': notes
            })
            # After updating balance in any trade (stop-loss, take-profit, reversal, close, etc.)
            # Insert this liquidation check after each place where balance is updated and before the next trade logic or end of loop
            if balance <= 0:
                notes = "LIQUIDATION: Balance <= 0, all positions closed."
                if position != 0:
                    close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
                    realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
                    realized_pnl *= leverage
                    fee = abs(position) * close_price * trading_fee_pct
                    balance += realized_pnl - fee
                    trade = {'date': row.name, 'action': 'LIQUIDATION', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
                    trades.append(trade)
                    position = 0
                    entry_price = 0
                observation_log.append({
                    'date': row.name,
                    'ticker': ticker,
                    'action': 'LIQUIDATION',
                    'quantity': 0,
                    'price': price,
                    'cash': balance,
                    'stock': 0,
                    'total_value': balance,
                    'realized_pnl': realized_pnl if 'realized_pnl' in locals() else 0,
                    'unrealized_pnl': 0,
                    'notes': notes
                })
                break

        # If still in position at end, close it
        if position != 0:
            price = df.iloc[-1]['close']
            close_price = price * (1 - slippage_pct - depth_slippage_factor * (abs(position) * price / avg_volume)) if position > 0 else price * (1 + slippage_pct + depth_slippage_factor * (abs(position) * price / avg_volume))
            realized_pnl = (close_price - entry_price) * position if position > 0 else (entry_price - close_price) * abs(position)
            realized_pnl *= leverage
            fee = abs(position) * close_price * trading_fee_pct
            balance += realized_pnl - fee
            trade = {'date': df.index[-1], 'action': 'CLOSE', 'price': close_price, 'quantity': position, 'pnl': realized_pnl - fee, 'balance': balance}
            trades.append(trade)
            if realized_pnl > 0:
                win_trades += 1
            else:
                loss_trades += 1
            if best_trade is None or realized_pnl > best_trade:
                best_trade = realized_pnl
            if worst_trade is None or realized_pnl < worst_trade:
                worst_trade = realized_pnl
            position = 0
            entry_price = 0
            equity_curve.append(balance)

        # 4. Results
        # Cap/filter extreme trade % for reporting
        min_trade_pnl = 0.001 * startup_capital  # 0.1% of starting capital
        filtered_trades = [t['pnl'] for t in trades if abs(t['pnl']) > min_trade_pnl]
        if filtered_trades:
            best_trade = max(filtered_trades)
            worst_trade = min(filtered_trades)
            # Optionally cap at ±500% of startup capital
            best_trade = min(best_trade, 5 * startup_capital)
            worst_trade = max(worst_trade, -5 * startup_capital)
        else:
            best_trade = 0
            worst_trade = 0
        total_return = (balance - startup_capital) / startup_capital
        win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
        num_trades = win_trades + loss_trades
        final_balance = balance
        summary_table = f"Return: {total_return*100:.2f}%, Max DD: {max_drawdown*100:.2f}%, Win Rate: {win_rate*100:.2f}%, Trades: {num_trades}"
        # Report best/worst trade as % of starting capital, or N/A if only one trade
        if num_trades <= 1:
            best_trade = worst_trade = None
        else:
            best_trade = (best_trade / startup_capital) * 100
            worst_trade = (worst_trade / startup_capital) * 100

        # 5. Export CSV
        csv_path = None

        # Action summary for Discord
        from collections import Counter
        action_counts = Counter([a['action'] for a in actions_log])
        sample_actions = actions_log[:3] + ([{'date': '...', 'action': '...'}] if len(actions_log) > 6 else []) + actions_log[-3:] if len(actions_log) > 6 else actions_log
        result = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "final_balance": final_balance,
            "best_trade": best_trade or 0,
            "worst_trade": worst_trade or 0,
            "summary_table": summary_table,
            "action_counts": dict(action_counts) if 'action_counts' in locals() else {},
            "sample_actions": sample_actions if 'sample_actions' in locals() else [],
            "consult_logs": consult_logs,
            "actions_log": actions_log,
            "equity_curve": equity_curve,
            "trades": trades,
            "observation_log": observation_log
        }
        return result, csv_path
    except Exception as e:
        logger.error(f"Backtest error: {e} | ticker={ticker}, timeframe={timeframe}, agents={agents}, observations={observations}, leverage={leverage}")
        raise

def save_backtest_results(result, ticker, timeframe, agent_name):
    """
    Save backtest results to a CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"backtest_{ticker}_{timeframe}_{timestamp}.csv"
    os.makedirs("backtest_results", exist_ok=True)
    filepath = os.path.join("backtest_results", filename)
    # Write observation log as main section
    with open(filepath, 'w', newline='') as f:
        fieldnames = ['date', 'action', 'price', 'quantity', 'realized_pnl', 'unrealized_pnl', 'balance']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obs in result.get('observation_log', []):
            writer.writerow(obs)
        # Write trades section
        writer.writerow({})
        writer.writerow({'date': 'TRADES'})
        trade_fields = ['date', 'action', 'price', 'quantity', 'pnl', 'balance']
        trade_writer = csv.DictWriter(f, fieldnames=trade_fields)
        trade_writer.writeheader()
        for trade in result.get('trades', []):
            trade_writer.writerow(trade)
        # Write consult outputs
        writer.writerow({})
        writer.writerow({'date': 'CONSULT_OUTPUTS'})
        consult_fields = ['date', 'consult_output']
        consult_writer = csv.DictWriter(f, fieldnames=consult_fields)
        consult_writer.writeheader()
        for consult in result.get('consult_logs', []):
            consult_writer.writerow({
                'date': consult['date'],
                'consult_output': str(consult['consult_output'])
            })
        # Write equity curve
        writer.writerow({})
        writer.writerow({'date': 'EQUITY_CURVE'})
        for i, eq in enumerate(result.get('equity_curve', [])):
            writer.writerow({'date': str(i), 'balance': eq})
    return filepath 