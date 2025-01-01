import numpy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)



class RsiquiV2(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = '5m'
    stoploss = -0.05
    trailing_stop = False
    max_open_trades = 10

    minimal_roi = {
      '0': 0.21000000000000002,
      '10': 0.042,
      '70': 0.028,
      '152': 0
    }

    rsi_entry_long  = IntParameter(0,  50,  default=50, space='buy',  optimize=True)
    rsi_entry_short = IntParameter(50, 100, default=50, space='buy',  optimize=True)
    rsi_exit_long   = IntParameter(50, 100, default=95, space='sell', optimize=True)
    rsi_exit_short  = IntParameter(0,  50,  default=43, space='sell', optimize=True)

    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi': {},
        }
        plot_config['subplots'] = {
            'Misc': {
                'rsi_gra' : {},
            },
        }

        return plot_config
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_gra'] = numpy.gradient(dataframe['rsi'], 60)
        
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

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0