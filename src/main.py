import os
import time
import logging
from datetime import datetime
from config.binance_config import BinanceConfig
from data.binance_provider import BinanceDataProvider
from tools.binance_executor import BinanceExecutor
from agents.binance_strategy import BinanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config() -> BinanceConfig:
    """Load configuration from environment variables."""
    config = BinanceConfig(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_API_SECRET'),
        trading_pairs=os.getenv('TRADING_PAIRS', '').split(',') if os.getenv('TRADING_PAIRS') else None,
        base_currency=os.getenv('BASE_CURRENCY', 'USDT'),
        quote_currency=os.getenv('QUOTE_CURRENCY', 'USDT'),
        default_order_type=os.getenv('DEFAULT_ORDER_TYPE', 'LIMIT'),
        default_time_in_force=os.getenv('DEFAULT_TIME_IN_FORCE', 'GTC'),
        max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.1')),
        stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', '0.02')),
        take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', '0.04')),
        candle_interval=os.getenv('CANDLE_INTERVAL', '1h'),
        update_interval=int(os.getenv('UPDATE_INTERVAL', '300')),  # 5 minutes in seconds
        rsi_period=int(os.getenv('RSI_PERIOD', '14')),
        rsi_overbought=float(os.getenv('RSI_OVERBOUGHT', '70')),
        rsi_oversold=float(os.getenv('RSI_OVERSOLD', '30')),
        macd_fast=int(os.getenv('MACD_FAST', '12')),
        macd_slow=int(os.getenv('MACD_SLOW', '26')),
        macd_signal=int(os.getenv('MACD_SIGNAL', '9'))
    )
    return config

def main():
    """Main function to run the trading bot."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize components
        data_provider = BinanceDataProvider(config.api_key, config.api_secret)
        executor = BinanceExecutor(config.api_key, config.api_secret)
        strategy = BinanceStrategy(config, data_provider, executor)
        
        logger.info("Trading bot started")
        logger.info(f"Trading pairs: {config.trading_pairs}")
        logger.info(f"Base currency: {config.base_currency}")
        logger.info(f"Quote currency: {config.quote_currency}")
        
        while True:
            try:
                # Run strategy for all symbols
                strategy.run_all_symbols()
                
                # Wait for next update
                time.sleep(config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
