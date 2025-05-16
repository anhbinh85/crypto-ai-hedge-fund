import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools
import logging

from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.analysts import ANALYST_ORDER
from src.main import run_hedge_fund
from src.tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from src.utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable
from src.utils.ollama import ensure_ollama_and_model
from config.binance_config import BinanceConfig
from data.binance_provider import BinanceDataProvider
from agents.binance_strategy import BinanceStrategy
from agents.trend_following_agent import TrendFollowingAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.portfolio_manager import PortfolioManagerCryptoAgent
from agents.risk_manager import RiskManagerCryptoAgent
from agents.sentiment import SentimentCryptoAgent
from agents.technicals import TechnicalsCryptoAgent
from agents.valuation import ValuationCryptoAgent
from agents.fundamentals import FundamentalsCryptoAgent
from agents.bill_ackman import BillAckmanCryptoAgent
from agents.peter_lynch import PeterLynchCryptoAgent
from agents.charlie_munger import CharlieMungerCryptoAgent
from agents.phil_fisher import PhilFisherCryptoAgent
from agents.michael_burry import MichaelBurryCryptoAgent
from agents.stanley_druckenmiller import StanleyDruckenmillerCryptoAgent
from agents.cathie_wood import CathieWoodCryptoAgent
from agents.warren_buffett import WarrenBuffettCryptoAgent
from agents.ben_graham import BenGrahamCryptoAgent

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        config: BinanceConfig,
        data_provider: BinanceDataProvider,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float
    ):
        """Initialize the backtester."""
        self.config = config
        self.data_provider = data_provider
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.logger = logging.getLogger(__name__)
        
        # Initialize results storage
        self.results: Dict[str, Dict] = {}
        
    def run_backtest(
        self,
        symbol: str,
        strategy_class: type,
        strategy_name: str
    ) -> dict:
        """Run backtest for a single strategy on a symbol."""
        try:
            # Get historical data
            df = self.data_provider.get_historical_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                start_time=self.start_date,
                end_time=self.end_date
            )
            
            # Initialize strategy
            strategy = strategy_class(
                config=self.config,
                data_provider=self.data_provider,
                executor=None  # No executor needed for backtesting
            )
            
            # If the agent has calculate_indicators/generate_signals, use the old loop
            has_signals = hasattr(strategy, 'calculate_indicators') and hasattr(strategy, 'generate_signals')
            balance = self.initial_balance
            position = None
            trades = []
            equity_curve = [balance]
            
            if has_signals:
                for i in range(len(df)):
                    current_data = df.iloc[:i+1]
                    current_price = current_data.iloc[-1]['close']
                    current_data = strategy.calculate_indicators(current_data)
                    buy_signal, sell_signal = strategy.generate_signals(current_data)
                    if position:
                        sl_tp_signal = strategy.check_stop_loss_take_profit(symbol, current_price)
                        if sl_tp_signal:
                            profit = (current_price - position['entry_price']) * position['quantity']
                            balance += profit
                            trades.append({
                                'entry_time': position['entry_time'],
                                'exit_time': current_data.index[-1],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'quantity': position['quantity'],
                                'profit': profit
                            })
                            position = None
                    if buy_signal and not position:
                        quantity = strategy.calculate_position_size(symbol, current_price)
                        if quantity > 0:
                            position = {
                                'entry_price': current_price,
                                'quantity': quantity,
                                'entry_time': current_data.index[-1]
                            }
                    elif sell_signal and position:
                        profit = (current_price - position['entry_price']) * position['quantity']
                        balance += profit
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_data.index[-1],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'quantity': position['quantity'],
                            'profit': profit
                        })
                        position = None
                    current_equity = balance
                    if position:
                        current_equity += (current_price - position['entry_price']) * position['quantity']
                    equity_curve.append(current_equity)
            else:
                # For simple agents, simulate buy-and-hold if their run_strategy would buy
                if len(df) == 0:
                    return None
                entry_price = df.iloc[0]['close']
                exit_price = df.iloc[-1]['close']
                # Simulate a single buy at the start if the agent would buy
                if hasattr(strategy, 'is_undervalued') and strategy.is_undervalued(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'is_sentiment_positive') and strategy.is_sentiment_positive(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'is_technical_signal_positive') and strategy.is_technical_signal_positive(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_strong_fundamentals') and strategy.has_strong_fundamentals(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_catalyst_and_risk_control') and strategy.has_catalyst_and_risk_control(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_growth_and_familiarity') and strategy.has_growth_and_familiarity(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_simplicity_and_incentives') and strategy.has_simplicity_and_incentives(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_innovation_and_leadership') and strategy.has_innovation_and_leadership(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'is_contrarian_and_undervalued') and strategy.is_contrarian_and_undervalued(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_macro_trend_and_conviction') and strategy.has_macro_trend_and_conviction(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'is_disruptive_and_high_growth') and strategy.is_disruptive_and_high_growth(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                elif hasattr(strategy, 'has_quality_and_moat') and strategy.has_quality_and_moat(symbol):
                    quantity = balance / entry_price
                    profit = (exit_price - entry_price) * quantity
                    trades.append({
                        'entry_time': df.index[0],
                        'exit_time': df.index[-1],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit': profit
                    })
                    balance += profit
                    equity_curve = [self.initial_balance, balance]
                else:
                    # No buy signal, do nothing
                    equity_curve = [self.initial_balance, self.initial_balance]
            # Calculate performance metrics
            trades_df = pd.DataFrame(trades)
            if len(trades_df) > 0:
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                total_profit = trades_df['profit'].sum()
                avg_profit = trades_df['profit'].mean()
                max_profit = trades_df['profit'].max()
                max_loss = trades_df['profit'].min()
                
                # Calculate drawdown
                equity_curve = np.array(equity_curve)
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (peak - equity_curve) / peak
                max_drawdown = np.max(drawdown)
                
                # Calculate Sharpe ratio
                returns = pd.Series(equity_curve).pct_change().dropna()
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
                
                results = {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'final_balance': balance,
                    'return_pct': (balance - self.initial_balance) / self.initial_balance * 100
                }
            else:
                results = {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'final_balance': balance,
                    'return_pct': 0
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest for {strategy_name} on {symbol}: {e}")
            return None

    def run_all_strategies(self) -> None:
        """Run backtests for all strategies on all symbols."""
        strategies = {
            'RSI_MACD': BinanceStrategy,
            'Trend_Following': TrendFollowingAgent,
            'Mean_Reversion': MeanReversionAgent,
            'Portfolio_Manager': PortfolioManagerCryptoAgent,
            'Risk_Manager': RiskManagerCryptoAgent,
            'Sentiment': SentimentCryptoAgent,
            'Technicals': TechnicalsCryptoAgent,
            'Valuation': ValuationCryptoAgent,
            'Fundamentals': FundamentalsCryptoAgent,
            'Bill_Ackman': BillAckmanCryptoAgent,
            'Peter_Lynch': PeterLynchCryptoAgent,
            'Charlie_Munger': CharlieMungerCryptoAgent,
            'Phil_Fisher': PhilFisherCryptoAgent,
            'Michael_Burry': MichaelBurryCryptoAgent,
            'Stanley_Druckenmiller': StanleyDruckenmillerCryptoAgent,
            'Cathie_Wood': CathieWoodCryptoAgent,
            'Warren_Buffett': WarrenBuffettCryptoAgent,
            'Ben_Graham': BenGrahamCryptoAgent
        }
        
        for symbol in self.config.trading_pairs:
            self.logger.info(f"Running backtests for {symbol}")
            for strategy_name, strategy_class in strategies.items():
                results = self.run_backtest(symbol, strategy_class, strategy_name)
                if results:
                    key = f"{symbol}_{strategy_name}"
                    self.results[key] = results

    def print_results(self) -> None:
        """Print backtest results in a formatted table."""
        if not self.results:
            self.logger.info("No backtest results available.")
            return
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # Format numeric columns
        numeric_cols = ['win_rate', 'total_profit', 'avg_profit', 'max_profit', 
                       'max_loss', 'max_drawdown', 'sharpe_ratio', 'return_pct']
        for col in numeric_cols:
            results_df[col] = results_df[col].map('{:.2f}'.format)
        
        # Print results
        print("\nBacktest Results:")
        print("=" * 100)
        print(results_df.to_string())
        print("=" * 100)
        
        # Print summary statistics
        print("\nStrategy Performance Summary:")
        print("-" * 50)
        summary = results_df.groupby('strategy').agg({
            'total_trades': 'sum',
            'win_rate': 'mean',
            'total_profit': 'sum',
            'return_pct': 'mean',
            'sharpe_ratio': 'mean'
        })
        print(summary.to_string())


### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument(
        "--tickers",
        type=str,
        required=False,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Margin ratio for short positions, e.g. 0.5 for 50% (default: 0.0)",
    )
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else []

    # Choose analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: " f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}")

    # Select LLM model based on whether Ollama is being used
    model_choice = None
    model_provider = None

    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

        # Select from Ollama-specific models
        model_choice = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        # Ensure Ollama is installed, running, and the model is available
        if not ensure_ollama_and_model(model_choice):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    else:
        # Use the standard cloud-based LLM selection
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            model_info = get_model_info(model_choice)
            if model_info:
                model_provider = model_info.provider.value
                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
            else:
                model_provider = "Unknown"
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create and run the backtester
    backtester = Backtester(
        config=BinanceConfig(
            trading_pairs=tickers,
            initial_balance=args.initial_capital,
            max_position_size=0.5,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            candle_interval="1h"
        ),
        data_provider=BinanceDataProvider(),
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
        initial_balance=args.initial_capital
    )

    backtester.run_all_strategies()
    backtester.print_results()
