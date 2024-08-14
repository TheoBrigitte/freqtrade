# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open

# BASED ON SMAOffset by Tirail.
# MOD BY Czaruś
# Backtested with 1 max open trade on Binance with USDT pairs, 5m timeframe.
# This strat trades once a day or once every other day.
#
#
# ======================================================= SELL REASON STATS ========================================================
# |        Sell Reason |   Sells |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |      39 |     39     0     0   100 |           3.49 |         136.09 |          28862.4  |         136.09 |
# |                roi |       1 |      1     0     0   100 |           8.99 |           8.99 |           1977.06 |           8.99 |
# ====================================================== LEFT OPEN TRADES REPORT ======================================================
# |   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |--------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# |  TOTAL |      0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
# ================== SUMMARY METRICS ===================
# | Metric                 | Value                     |
# |------------------------+---------------------------|
# | Backtesting from       | 2021-04-30 00:00:00       |
# | Backtesting to         | 2021-07-27 08:35:00       |
# | Max open trades        | 1                         |
# |                        |                           |
# | Total/Daily Avg Trades | 40 / 0.45                 |
# | Starting balance       | 10000.000 USDT            |
# | Final balance          | 40839.439 USDT            |
# | Absolute profit        | 30839.439 USDT            |
# | Total profit %         | 308.39%                   |
# | Avg. stake amount      | 19858.310 USDT            |
# | Total trade volume     | 794332.392 USDT           |
# |                        |                           |
# | Best Pair              | DATA/USDT 20.44%          |
# | Worst Pair             | ADA/USDT 0.0%             |
# | Best trade             | ONG/USDT 8.99%            |
# | Worst trade            | ETC/USDT 2.2%             |
# | Best day               | 5198.630 USDT             |
# | Worst day              | 0.000 USDT                |
# | Days win/draw/lose     | 18 / 42 / 0               |
# | Avg. Duration Winners  | 11:22:00                  |
# | Avg. Duration Loser    | 0:00:00                   |
# | Rejected Buy signals   | 536137                    |
# |                        |                           |
# | Min balance            | 0.000 USDT                |
# | Max balance            | 0.000 USDT                |
# | Drawdown               | 0.0%                      |
# | Drawdown               | 0.000 USDT                |
# | Drawdown high          | 0.000 USDT                |
# | Drawdown low           | 0.000 USDT                |
# | Drawdown Start         | 1970-01-01 00:00:00+00:00 |
# | Drawdown End           | 1970-01-01 00:00:00+00:00 |
# | Market change          | -53.09%                   |
# ======================================================



low_offset = 0.958 # something lower than 1
high_offset = 1.012 # something higher than 1


class BigTrader(IStrategy):
    INTERFACE_VERSION = 2
    # ROI table:
    minimal_roi = {
        "0": 0.09,
    }

    # Stoploss:
    stoploss = -0.5

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.029
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
#    def informative_pairs(self):
        # get access to all pairs available in whitelist.
#        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
#        informative_pairs = [("BTC/USDT", "5m")
#                            ]
#        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
#        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe)

        # SMA
#        informative['sma_10'] = ta.SMA(informative, timeperiod=10)
#        informative['sma_4'] = ta.SMA(informative, timeperiod=4)

#        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '5m', ffill=True)
        
#        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
#        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
#        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
#        dataframe['sma_3'] = ta.SMA(dataframe, timeperiod=3)
#        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
#        dataframe['sma_10'] = ta.SMA(dataframe, timeperiod=10)
#        dataframe['volume_shifted'] = dataframe['volume'].shift(3)
#        dataframe['volume_shifted_sold'] = dataframe['volume'].shift(4)
#        dataframe['volume_shifted_buy'] = dataframe['volume'].shift(1)
#        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < (dataframe['sma_15'] * low_offset))
                &
                (dataframe['close'] > dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(8) > dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(12) > dataframe['close'].shift(8))
                &
                (dataframe['volume'] > 0)
                ),
            'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] > (dataframe['sma_15'] * high_offset))
                &
                (dataframe['open'] < dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(8) < dataframe['close'].shift(4))
                &
                (dataframe['close'].shift(12) < dataframe['close'].shift(8))
                &
                (dataframe['volume'] > 0)
                ),
            'sell'] = 1
        return dataframe