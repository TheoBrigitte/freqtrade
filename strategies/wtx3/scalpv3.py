import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import datetime

class WTX3(IStrategy):
    INTERFACE_VERSION = 3
    # 定义交易对的时间框架
    timeframe = '5m'
    # 定义最小ROI的设置
    minimal_roi = {'0': 0.06, '5': 0.055, '10': 0.04, '15': 0.03, '20': 0.02, '25': 0.01}
    stoploss = -0.7  # 定义固定的止损
    trailing_stop = True  # 启用追踪止损
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.045
    trailing_only_offset_is_reached = True
    can_short = True
    exit_profit_only = False
    # 定义可调参数
    n1 = IntParameter(5, 20, default=10, space='buy')
    n2 = IntParameter(5, 30, default=21, space='buy')
    moneyFlowMultiplier = DecimalParameter(1, 10, default=5, space='buy')
    moneyFlowMultiplierSlow = DecimalParameter(1, 10, default=5, space='buy')
    # 添加绘图配置
    plot_config = {'main_plot': {'wt1': {'color': 'green', 'title': 'WaveTrend 1 (WT1)'}, 'wt2': {'color': 'red', 'title': 'WaveTrend 2 (WT2)'}, 'fast_money_flow': {'color': 'blue', 'title': 'Fast Money Flow'}, 'slow_money_flow': {'color': 'yellow', 'title': 'Slow Money Flow'}}}

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        return 10.0  # Apply 10x leverage

    def smoothrng(self, dataframe: DataFrame, period: int, multiplier: float) -> pd.Series:
        abs_diff = abs(dataframe['close'] - dataframe['close'].shift(1))
        smooth_range = ta.EMA(abs_diff, timeperiod=period)
        smooth_range = ta.EMA(smooth_range, timeperiod=period * 2 - 1) * multiplier
        return pd.Series(smooth_range, index=dataframe.index)

    def range_filter(self, dataframe: DataFrame, smooth_range: pd.Series) -> pd.Series:
        rngfilt = dataframe['close'].copy()
        for i in range(1, len(dataframe)):
            if rngfilt.iloc[i - 1] is not None:
                if dataframe['close'].iloc[i] > rngfilt.iloc[i - 1]:
                    rngfilt.iloc[i] = max(dataframe['close'].iloc[i] - smooth_range.iloc[i], rngfilt.iloc[i - 1])
                else:
                    rngfilt.iloc[i] = min(dataframe['close'].iloc[i] + smooth_range.iloc[i], rngfilt.iloc[i - 1])
        return rngfilt

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加指标到市场数据
        """
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['esa'] = ta.EMA(dataframe['hlc3'], timeperiod=self.n1.value)
        dataframe['d'] = ta.EMA(abs(dataframe['hlc3'] - dataframe['esa']), timeperiod=self.n1.value)
        dataframe['ci'] = (dataframe['hlc3'] - dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['tci'] = ta.EMA(dataframe['ci'], timeperiod=self.n2.value)
        dataframe['wt1'] = dataframe['tci']
        dataframe['wt2'] = ta.SMA(dataframe['wt1'], timeperiod=4)
        # 添加 money flow 指标
        dataframe['fast_money_flow'] = 2 * ta.SMA(dataframe['hlc3'] - ta.SMA(dataframe['hlc3'], 9), 9) / ta.SMA(dataframe['high'] - dataframe['low'], 9) * self.moneyFlowMultiplier.value
        dataframe['slow_money_flow'] = 2 * ta.SMA(dataframe['hlc3'] - ta.SMA(dataframe['hlc3'], 10), 10) / ta.SMA(dataframe['high'] - dataframe['low'], 10) * self.moneyFlowMultiplierSlow.value
        # 添加Range Filter指标
        smooth_range = self.smoothrng(dataframe, 100, 3.0)
        dataframe['rngfilt'] = self.range_filter(dataframe, smooth_range)
        dataframe['hband'] = dataframe['rngfilt'] + smooth_range
        dataframe['lband'] = dataframe['rngfilt'] - smooth_range
        # 添加RSI指标
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=14)
        # 添加MACD指标
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'], fastperiod=14, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macdsignal
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义开仓条件
        """
        # 检测多头交叉点
        dataframe['cross_above'] = ((dataframe['wt1'] > dataframe['wt2']) & (dataframe['wt1'].shift() <= dataframe['wt2'].shift())).astype(int)
        dataframe['cross_below'] = ((dataframe['wt1'] < dataframe['wt2']) & (dataframe['wt1'].shift() >= dataframe['wt2'].shift())).astype(int)
        # 多头开仓条件
        dataframe.loc[(dataframe['cross_above'] == 1) & (dataframe['fast_money_flow'] > dataframe['slow_money_flow']) & (dataframe['close'] < dataframe['hband']) & (dataframe['rsi'] < 70) & (dataframe['macd'] > dataframe['macd_signal']), 'enter_long'] = 1
        # 空头开仓条件
        dataframe.loc[(dataframe['cross_below'] == 1) & (dataframe['fast_money_flow'] < dataframe['slow_money_flow']) & (dataframe['close'] > dataframe['lband']) & (dataframe['rsi'] > 30) & (dataframe['macd'] < dataframe['macd_signal']), 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            定义平仓条件
            """
        # 检测多头交叉点
        dataframe['cross_below'] = ((dataframe['wt1'] < dataframe['wt2']) & (dataframe['wt1'].shift() >= dataframe['wt2'].shift())).astype(int)
        dataframe['cross_above'] = ((dataframe['wt1'] > dataframe['wt2']) & (dataframe['wt1'].shift() <= dataframe['wt2'].shift())).astype(int)
        # 多头平仓条件
        dataframe.loc[(dataframe['cross_below'] == 1) & (dataframe['close'] > dataframe['hband']) & (dataframe['rsi'] > 70) & (dataframe['macd'] < dataframe['macd_signal']), 'exit_long'] = 1
        # 空头平仓条件
        dataframe.loc[(dataframe['cross_above'] == 1) & (dataframe['close'] < dataframe['lband']) & (dataframe['rsi'] < 30) & (dataframe['macd'] > dataframe['macd_signal']), 'exit_short'] = 1
        return dataframe