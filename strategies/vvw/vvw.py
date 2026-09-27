
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import pandas as pd
import numpy as np
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

#def VolatilityOscillator(dataframe, window_size=100, num_of_std=1):
#    df = dataframe.copy()
#    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
#    rolling_std = df['vwap'].rolling(window=window_size).std()
#    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
#    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
#
#    dataframe["vo_spike"] = (dataframe["close"] - dataframe["open"])
#    dataframe["vo_x"] = ta.STDDEV(price='vo_spike', timeperiod=window_size)
#    dataframe["vo_y"] = (ta.STDDEV(price='vo_spike', timeperiod=window_size) * -1)
#
#    return df['vwap_low'], df['vwap'], df['vwap_high']

# ChatGPT version
def calculate_volatility_oscillator_gpt(dataframe, length=100, lbR=5, lbL=5, rangeUpper=60, rangeLower=5):
    data = dataframe.copy()

    close = data['close']
    open_ = data['open']
    low = data['low']

    # Calculate Spike
    spike = close - open_

    # Standard Deviation calculations using TA-Lib
    x = ta.STDDEV(spike, timeperiod=length)
    y = x * -1

    #def pivot_low(series, lbL, lbR):
    #    """
    #    Identifies pivot lows in a time series.

    #    Parameters:
    #        series (pd.Series): The input series (e.g., price low values).
    #        lbL (int): Number of bars to the left to check.
    #        lbR (int): Number of bars to the right to check.

    #    Returns:
    #        pd.Series: A boolean series indicating pivot lows.
    #    """
    #    # Initialize an array to store pivot low flags
    #    pivots = np.full(len(series), False)

    #    # Loop through the series, skipping the boundaries
    #    for i in range(lbL, len(series) - lbR):
    #        left_condition = all(series[i] < series[i - j] for j in range(1, lbL + 1))
    #        right_condition = all(series[i] < series[i + j] for j in range(1, lbR + 1))
    #        pivots[i] = left_condition and right_condition

    #    return pd.Series(pivots, index=series.index)

    # Helper functions for pivot points
    def pivot_low(series, lbL, lbR):
        pivots = [None] * len(series)  # Initialize with None
        for i in range(lbL, len(series) - lbR):
            if all(series[i] < series[i - j] for j in range(1, lbL + 1)) and all(series[i] < series[i + j] for j in range(1, lbR + 1)):
                pivots[i] = True
            else:
                pivots[i] = False
        return pd.Series(pivots, index=series.index)

    def checkhl(data_back, data_forward, hl):
        if hl == 'high' or hl == 'High':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref < data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref <= data_forward[i]:
                    return 0
            return 1
        if hl == 'low' or hl == 'Low':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref > data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref >= data_forward[i]:
                    return 0
            return 1

    def pivot(osc, LBL, LBR, highlow):
        left = []
        right = []
        pivots = pd.Series([0.0] * len(osc), index=osc.index)
        for i in range(len(osc)):
            pivots._append(0.0)
            if i < LBL + 1:
                left.append(osc[i])
            if i > LBL:
                right.append(osc[i])
            if i > LBL + LBR:
                left.append(right[0])
                left.pop(0)
                right.pop(0)
                if checkhl(left, right, highlow):
                    pivots[i - LBR] = osc[i - LBR]
        return pivots

    def bars_since(series):
        last_true = None
        bars = []
        for i, val in enumerate(series):
            if val:
                last_true = i
            bars.append(i - last_true if last_true is not None else np.nan)
        return pd.Series(bars, index=series.index)

    def in_range(cond):
        bars = bars_since(cond)
        return (bars >= rangeLower) & (bars <= rangeUpper)

    # Identify pivot lows
    pl_found = pivot_low(spike, lbL, lbR)

    # Debug pivot lows
    print("pl_found :", pl_found.iat[-1])

    # Oscillator: Higher Low
    prev_pivot_spike = spike.shift(lbR).where(pl_found.shift(1))
    #print("Previous Pivot Spike (prev_pivot_spike):", prev_pivot_spike.dropna().head())

    osc_hl = (spike.shift(lbR) > prev_pivot_spike) & in_range(pl_found.shift(1))
    print("osc_hl   :", osc_hl.iat[-1])

    # Price: Lower Low
    prev_pivot_low = low.shift(lbR).where(pl_found.shift(1))
    #print("Previous Pivot Low (prev_pivot_low):", prev_pivot_low.dropna().head())

    price_ll = (low.shift(lbR) < prev_pivot_low)
    print("price_ll :", price_ll.iat[-1])

    # Bullish Condition
    bull_cond = price_ll & osc_hl & pl_found

    # Debug final bullish condition
    print("bull_cond:", bull_cond.iat[-1])

    return {
        'spike': spike,
        'upper_line': x,
        'lower_line': y,
        'bull_cond': bull_cond
    }

def calculate_volatility_oscillator_seek(data, length=100, lbR=5, lbL=5, rangeUpper=60, rangeLower=5):
    # Calculate spike (close - open)
    data['spike'] = data['close'] - data['open']

    # Standard deviation calculations
    data['upper_line'] = data['spike'].rolling(window=length).std()
    data['lower_line'] = -data['upper_line']

    # Pivot Low Logic
    def pivot_low(series, lbL, lbR):
        pivot_lows = []
        for i in range(lbL, len(series) - lbR):
            window = series.iloc[i - lbL : i + lbR + 1]
            if series.iloc[i] == window.min():
                pivot_lows.append(series.iloc[i])
            else:
                pivot_lows.append(None)
        # Align with original data length
        return pd.Series([None]*lbL + pivot_lows + [None]*lbR, index=series.index)

    data['osc2'] = data['spike']
    data['pivot_low'] = pivot_low(data['osc2'], lbL, lbR)
    data['plFound'] = data['pivot_low'].notna()

    # Helper function: _inRange
    #def in_range(cond, rangeLower, rangeUpper):
    #    bars_since = cond[::-1].cumsum()[::-1]  # Reverse to count bars since condition
    #    return (bars_since >= rangeLower) & (bars_since <= rangeUpper)

    def bars_since(series):
        last_true = None
        bars = []
        for i, val in enumerate(series):
            if val:
                last_true = i
            bars.append(i - last_true if last_true is not None else np.nan)
        return pd.Series(bars, index=series.index)

    def in_range(cond, rangeLower, rangeUpper):
        bars = bars_since(cond)
        return (bars >= rangeLower) & (bars <= rangeUpper)

    # Bullish Conditions
    data['oscHL'] = (data['osc2'].shift(lbR) > data['osc2'].shift(lbR).where(data['plFound'].shift(1)).ffill()) & \
                    in_range(data['plFound'].shift(1), rangeLower, rangeUpper)

    data['priceLL'] = data['low'].shift(lbR) < data['low'].shift(lbR).where(data['plFound'].shift(1)).ffill()
    data['bull_cond'] = data['priceLL'] & data['oscHL'] & data['plFound']

    # Resulting DataFrame now contains all calculated columns:
    # spike, x, y, osc2, pivot_low, plFound, oscHL, priceLL, bullCond
    return data

class VVW(IStrategy):
    """
    VVW strategy
    author@: Théo Brigitte
    github@: https://github.com/TheoBrigitte

    Strategy based on Volatility Oscillator and VWAP
    """

    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, window_size=15, num_of_std=1)
        dataframe['vwap_low'] = vwap_low
        dataframe['vwap'] = vwap
        dataframe['vwap_high'] = vwap_high

        #volatility_oscillator = calculate_volatility_oscillator_gpt(dataframe, length=100, lbR=5, lbL=5, rangeUpper=60, rangeLower=5)
        volatility_oscillator = calculate_volatility_oscillator_seek(dataframe, length=20, lbR=5, lbL=5, rangeUpper=60, rangeLower=5)
        dataframe['volatility_upper'] = volatility_oscillator['upper_line']
        dataframe['volatility_lower'] = volatility_oscillator['lower_line']
        dataframe['volatility_spike'] = volatility_oscillator['spike']
        dataframe['volatility_bullish'] = volatility_oscillator['bull_cond']
        dataframe['oscHL'] = volatility_oscillator['oscHL']
        dataframe['priceLL'] = volatility_oscillator['priceLL']
        dataframe['plFound'] = volatility_oscillator['plFound']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                dataframe['volatility_bullish'] &
                (dataframe['rsi'] < 37) &
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        #dataframe.loc[
        #    (
        #        qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100']) &
        #        (dataframe['ha_close'] < dataframe['ema20']) &
        #        (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
        #    ),
        #    'exit_long'] = 1

        #dataframe.loc[
        #    (
        #        qtpylib.crossed_above(dataframe['close'], dataframe['vwap_high']) &
        #        (dataframe['volatility_bullish'] == False)
        #    ),
        #    'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > 68) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe

    @property
    def plot_config(self):
        """
            There are a lot of solutions how to build the return dictionary.
            The only important point is the return value.
            Example:
                plot_config = {'main_plot': {}, 'subplots': {}}

        """
        plot_config = {}
        plot_config['main_plot'] = {
            'vwap_high': {'color': 'blue', 'fill_to': 'vwap_low'},
            'vwap': {'color': 'red'},
            'vwap_low': {'color': 'blue'}
        }
        plot_config['subplots'] = {
            "RSI": {'rsi': {}},
            "VolatilityOscillator": {
                'volatility_upper': {'color': 'blue', 'fill_to': 'volatility_lower'},
                'volatility_spike': {'color': 'orange'},
                'volatility_lower': {'color': 'blue'}
            },
            "VolatilityBullish": {
                'volatility_bullish': {'color': 'green'},
                #'oscHL': {'color': 'blue'},
                #'priceLL': {'color': 'red'},
                #'plFound': {'color': 'yellow'}
            },
        }

        return plot_config
