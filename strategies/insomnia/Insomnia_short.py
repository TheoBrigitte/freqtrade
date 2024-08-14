# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import math
import numpy as np  # noqa
import pandas as pd  # noqa
import talib as tlb
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union, Tuple
from freqtrade.persistence import Trade
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter, informative)


class Insomnia_short(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = True
    levarage_input = 10.0
    stoploss = -0.01
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.04
    position_adjustment_enable = True
    timeframe = "1h"
    minimal_roi = {
        "0": 0.085,
        "128": 0.0259,
        "165": 0.08,
    }
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    buy_params = {
        "buy_rsi": 14,
        "buy_rsi_compare_long": 30,
        "buy_rsi_compare_short": 68,
    }
    sell_params = {
        "sell_rsi_compare_long": 70,
        "sell_rsi_compare_short": 28,
        "sell_decr_pos": 0.16
    }

    buy_rsi = IntParameter(8, 20, default=buy_params['buy_rsi'], space="buy", optimize=False)
    buy_rsi_compare_long = IntParameter(20, 65, default=buy_params['buy_rsi_compare_long'], space="buy", optimize=True)
    buy_rsi_compare_short = IntParameter(20, 65, default=buy_params['buy_rsi_compare_short'], space="buy",
                                         optimize=False)
    sell_rsi_compare_long = IntParameter(60, 80, default=sell_params['sell_rsi_compare_long'], space="sell",
                                         optimize=False)
    sell_rsi_compare_short = IntParameter(20, 40, default=sell_params['sell_rsi_compare_short'], space="sell",
                                          optimize=False)

    long_multiplier = DecimalParameter(0.95, 1.0, default=0.998, decimals=3, space='buy', optimize=True)
    short_multiplier_1 = DecimalParameter(1.0, 1.05, default=1.01, decimals=3, space='buy', optimize=True)

    buy_factor = DecimalParameter(1, 5, default=3, decimals=1, space="buy", optimize=False)
    buy_period = IntParameter(4, 15, default=10, space="buy", optimize=False)

    sell_decr_pos = DecimalParameter(0.01, 0.3, decimals=3, default=sell_params['sell_decr_pos'], optimize=True)

    startup_candle_count: int = 400
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val1 in self.buy_factor.range:
            for val2 in self.buy_period.range:
                short_str = f'{val1}{val2}'
                dataframe[['direction_bigger' + short_str, 'Final_ub' + short_str]] = \
                    self.supertrend(dataframe, multiplier=val1, period=val2)[
                        ['STX', 'Fin_band']]
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["adx"] = ta.ADX(dataframe)
        for val in self.buy_rsi.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        dataframe['WMA1'] = tlb.WMA(dataframe["close"], 60 / 2) * 2
        dataframe['WMA2'] = tlb.WMA(dataframe["close"], 60)
        dataframe['SSL'] = tlb.WMA(dataframe['WMA1'] - dataframe['WMA2'], round(math.sqrt(60)))
        ssl_down, ssl_up = SSLChannels(dataframe, 10)
        dataframe["ssl_down"] = ssl_down
        dataframe["ssl_up"] = ssl_up
        dataframe["ssl-dir"] = np.where(ssl_up > ssl_down, "up", "down")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_str = f'{self.buy_factor.value}{self.buy_period.value}'
        condition_long = (
                (qtpylib.crossed_below(dataframe['SSL'], dataframe['close'])) &
                (dataframe[f'direction_bigger{short_str}_4h'] == 'up')
            # (dataframe[f'rsi_{self.buy_rsi.value}'] < self.buy_rsi_compare_long.value)
            # (dataframe["ssl_up"] > dataframe["ssl_down"])
            # (((dataframe['high'] - dataframe['low']) / dataframe['low']) > 0.005)
        )
        dataframe.loc[condition_long, 'enter_long'] = 0
        dataframe.loc[condition_long, 'enter_tag'] = 'Long'

        condition_short = (
            (dataframe[f'direction_bigger{short_str}_4h'] == 'down') &
            (qtpylib.crossed_above(dataframe['SSL'], dataframe['close'])) &
            (dataframe["ssl_up"] < dataframe["ssl_down"])
            # (((dataframe['high'] - dataframe['low']) / dataframe['low']) > 0.005)
        )
        dataframe.loc[condition_short, 'enter_short'] = 1
        dataframe.loc[condition_short, 'enter_tag'] = 'Short'
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional["Trade"], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        entry_price = 0
        if side == "long":
            if entry_tag == 'Long':
                entry_price = dataframe['close'].iat[-1] * self.long_multiplier.value
        else:
            if entry_tag == 'Short':
                entry_price = dataframe['close'].iat[-1] * self.short_multiplier_1.value
        return entry_price

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['SSL'], dataframe['close'])),
            'exit_long'] = 0
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['SSL'], dataframe['close'])),
            'exit_short'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        if current_profit > self.sell_decr_pos.value and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 1.5), 'half_profit_5%'
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        if self.levarage_input > max_leverage:
            return max_leverage

        return self.levarage_input

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                             df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else \
                df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                             df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else \
                df['final_lb'].iat[i - 1]
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[
                i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > \
                                         df['final_ub'].iat[i] else \
                    df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= \
                                             df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < \
                                                 df['final_lb'].iat[i] else 0.00
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)
        df['final_band'] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), df['final_lb'], df['final_ub']),
                                    np.NaN)
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
        return DataFrame(index=df.index, data={
            'ST': df[st],
            'STX': df[stx],
            'Fin_band': df['final_band']
        })


def SSLChannels_ATR(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]


# SSL Channels
def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]
