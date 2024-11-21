# --- Required Libraries ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
import talib as ta
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter
import warnings
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import numpy as np


warnings.simplefilter(action="ignore", category=RuntimeWarning)


# Volume Weighted Moving Average
def VWMA(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma


def RMA(series: Series, length: int = 10):
    rma = series.copy()
    rma.iloc[:length] = rma.rolling(length).mean().iloc[:length]
    rma = rma.ewm(alpha=(1.0/length), adjust=False).mean()
    return rma


def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)


def hma(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_weighted_mean(series, max(1, math.floor(window / 2)), min_periods)) - rolling_weighted_mean(
        series, window, min_periods
    )
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)


def ma(dataframe: DataFrame, length, ma_type):
    if 'close' not in dataframe:
        raise KeyError("The 'close' column is missing from the dataframe.")

    if ma_type == "SMA":
        return ta.SMA(dataframe['close'], length)
    elif ma_type == "EMA":
        return ta.EMA(dataframe['close'], length)
    elif ma_type == "RMA":
        return RMA(dataframe['close'], length)
    elif ma_type == "HMA":
        return hma(dataframe['close'], length)
    elif ma_type == "WMA":
        return ta.WMA(dataframe['close'], length)
    elif ma_type == "VWMA":
        return VWMA(dataframe, length)


class JuicyTrend(IStrategy):
    INTERFACE_VERSION = 3

    plot_config = {
        'main_plot': {
            'ma1': {'color': '#FF5733'},
            'ma2': {'color': '#FF8333'},
            'ma3': {'color': '#FFB533'},
        },
        'subplots': {}
    }

    # Define the timeframes and settings
    timeframe = '15m'  # Set to '1m' for 1-minute timeframe
    stoploss = -0.03  # 8% stop loss
    minimal_roi = {"0": 0.11}  # 11% take profit equivalent to 10% in PineScript

    # --- Strategy Parameters ---

    ma1_type = CategoricalParameter(["SMA", "EMA", "RMA", "HMA", "WMA", "VWMA"], default="RMA", space='buy', optimize=True, load=True)
    ma2_type = CategoricalParameter(["SMA", "EMA", "RMA", "HMA", "WMA", "VWMA"], default="HMA", space='buy', optimize=True, load=True)
    ma3_type = CategoricalParameter(["SMA", "EMA", "RMA", "HMA", "WMA", "VWMA"], default="EMA", space='buy', optimize=True, load=True)

    tradetrendoption = BooleanParameter(default=False, space="buy", optimize=True)  # Trade only in trend direction
    sellaftertrend = BooleanParameter(default=False, space="sell", optimize=True)  # Sell After Trend Reverses # If you are only trading with the trend, there may be a position which remains open after that trend has ended./nThis option will force that final trade to close to limit the loss and get the strategy back into action on the next uptrend.
    # sellout = IntParameter(low=0, high=10000, default=10, space='buy', optimize=True, load=True)     # Sell After (bars)"
    # sellinprofitoption = BooleanParameter(default=False, space="sell", optimize=True)   # Only Sell in Profit

    ma1_length = IntParameter(low=1, high=20, default=7, space='buy', optimize=True, load=True) # \"MA 1\" should follow the price closely while reducing as much noise as possible.\n\"BUY\" and \"SELL\" signals are based on \"MA 1\" and \"MA 2\" crosses.\nAdjust them to flow with your asset and timeframe as profitably as possible.
    ma2_length = IntParameter(low=20, high=200, default=55, space='buy', optimize=True, load=True) # \"MA 2\" should not make contact with \"MA 1\" until the direction of price has started reversing.\nTry to keep the lines seperated, only crossing on pivot highs and pivot lows.
    ma3_length = IntParameter(low=200, high=600, default=300, space='buy', optimize=True, load=True) # If \"Only Trade with Trend\" is enabled, \"MA 3\" will restrict trades to it's direction.\n\"BUY\" trades will only be executed in an uptrend.

    usestochrsi = BooleanParameter(default=True, space="buy", optimize=True) # Decide whether to consider the position of the Stochastic RSI. If enabled, a "BUY" order will only be executed if the Stochastic RSI is below the decided level.
    stochrsilevel = IntParameter(low=5, high=95, default=60, space='buy', optimize=True, load=True) # A "BUY" order can only be executed if the Stochastic RSI is below this level.
    lengthRSI = IntParameter(low=7, high=30, default=16, space='buy', optimize=True, load=True)  # The price action of every asset is different, so the Stochastic RSI must be tuned and timed with the market.\nTaking the time to adjust this value will have a great impact on profitability.
    lengthStoch = IntParameter(low=10, high=50, default=20, space='buy', optimize=True, load=True)  # The Stochastic length should typically be slightly longer than the RSI length. After adjusting all settings, start from the top and do it again.\nRepeatedly tuning this indicator from top to bottom will give you the best results.

    # --- Populate Indicators ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Calculate Moving Averages
        dataframe['ma1'] = ma(dataframe=dataframe, length=self.ma1_length.value, ma_type=self.ma1_type.value)
        dataframe['ma2'] = ma(dataframe=dataframe, length=self.ma2_length.value, ma_type=self.ma2_type.value)
        dataframe['ma3'] = ma(dataframe=dataframe, length=self.ma3_length.value, ma_type=self.ma3_type.value)

        # Calculate Stochastic RSI
        smoothK = 3
        # rsi1 = ta.RSI(dataframe['close'], self.lengthRSI.value)
        # k = ta.SMA(ta.STOCH(rsi1, rsi1, rsi1, self.lengthStoch.value), smoothK)
        fastk, fastd = ta.STOCHRSI(dataframe['close'], self.lengthStoch.value, smoothK)
        dataframe['srsi_fk'] = fastk
        dataframe['srsi_fd'] = fastd

        return dataframe

    # --- Buy Signal ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy = dataframe['ma1'] > dataframe['ma2']
        buystochrsi = buy & dataframe['srsi_fk'] < self.stochrsilevel.value
        buywithtrend = buy & dataframe['ma3'] > dataframe['ma3'].shift(1)
        buystochrsitrue = buystochrsi if self.usestochrsi.value else buy
        buytrendtrue = buywithtrend if self.tradetrendoption.value else buy
        volume = (dataframe['volume'] > 0)  # Make sure Volume is not 0
        buyoption = buystochrsitrue & buytrendtrue & volume

        dataframe.loc[buyoption, 'enter_long'] = 1
        return dataframe

    # --- Sell Signal ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sell = dataframe['ma1'] < dataframe['ma2']
        close_above_ma3 = dataframe['close'] > dataframe['ma3'] if self.sellaftertrend.value else sell
        volume_positive = dataframe['volume'] > 0
        conditions = sell & volume_positive & close_above_ma3
        dataframe.loc[conditions, 'exit_long'] = 1
        return dataframe
