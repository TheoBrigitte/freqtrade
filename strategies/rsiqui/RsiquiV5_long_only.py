import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Tuple, Union
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)

class RsiquiV5_long_only(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = True

    # Buy hyperspace params:
    buy_params = {
        "rsi_entry_long": 27,
        "rsi_entry_short": 59,
        "window": 24,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsi_exit_long": 18,
        "rsi_exit_short": 75,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.223,
        "34": 0.082,
        "82": 0.033,
        "109": 0
    }

    # Stoploss:
    stoploss = -0.273

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy
    
    # Max Open Trades:
    max_open_trades = -1

    rsi_entry_long  = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    rsi_exit_long   = IntParameter(0, 100, default=buy_params.get('rsi_exit_long'),   space='sell', optimize=True)
    rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)
    rsi_exit_short  = IntParameter(0, 100, default=buy_params.get('rsi_exit_short'),  space='sell', optimize=True)
    window          = IntParameter(5, 100, default=buy_params.get('window'),          space='buy',  optimize=False)

    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi_ema' : {}
        }
        plot_config['subplots'] = {
            'Misc': {
                'rsi': {},
                'rsi_gra' : {},
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_entry_long.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_entry_short.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_exit_long.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_exit_short.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 10
