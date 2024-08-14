from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

import talib.abstract as ta
from talib import MA_Type
import freqtrade.vendor.qtpylib.indicators as qtpylib

class momentum(IStrategy):

    INTERFACE_VERSION = 3

    # Settings
    timeframe = "5m"
    startup_candle_count = 5
    can_short = True
    use_exit_signal = True

    # Take profit
    minimal_roi = {"0": 0.02}

    # Stoploss
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Strong uptrend
                #
                # Price action
                (dataframe['close'] > dataframe['open']) &
                (dataframe['close'] > dataframe['close'].shift(1)) &
                (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
                (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &
                (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &
                (dataframe['close'].shift(2) > dataframe['close'].shift(3)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (0, 'long_trend')

        dataframe.loc[
            (
                # Strong downtrend
                #
                # Price action
                (dataframe['close'] < dataframe['open']) &
                (dataframe['close'] < dataframe['close'].shift(1)) &
                (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
                (dataframe['close'].shift(1) < dataframe['close'].shift(2)) &
                (dataframe['close'].shift(2) < dataframe['open'].shift(2)) &
                (dataframe['close'].shift(2) < dataframe['close'].shift(3)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'short_trend')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
