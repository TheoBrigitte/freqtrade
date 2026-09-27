# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
# --------------------------------

class Rsiqui_protections(IStrategy):
    INTERFACE_VERSION = 3
    # Random ROI chosen
    minimal_roi = {'0': 0.1}
    # Random stoploss
    stoploss = -0.25
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # Calculates slope of the RSI
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi'], 60)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy signal generated when RSI lower than 30 and the slope becomes positive.
        dataframe.loc[(dataframe['rsi'] < 30) & qtpylib.crossed_above(dataframe['rsi_gra'], 0), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sell signal generated when RSI above 60 and the slope becomes negative.
        dataframe.loc[(dataframe['rsi'] > 60) & qtpylib.crossed_below(dataframe['rsi_gra'], 0), 'exit_long'] = 1
        return dataframe

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
