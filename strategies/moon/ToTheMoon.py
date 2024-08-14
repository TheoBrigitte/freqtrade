
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib
import ephem

class ToTheMoon(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.99

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    timeframe = '1h'

    can_short: bool = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['moon_phase'] = dataframe['date'].apply(lambda x: ephem.Moon(ephem.Date(x)).phase)
        # dataframe['days_since_fm'] = dataframe['date'].apply(lambda x: round(ephem.Date(x) - ephem.previous_full_moon(ephem.Date(x))))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                    qtpylib.crossed_below(dataframe['moon_phase'],99) # enter long just after full moon
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                    qtpylib.crossed_above(dataframe['moon_phase'],1) # enter short just after new moon
            ),
            'enter_short'] = 1
        

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['moon_phase'],1) # exit long just before new moon
            ),
            'exit_long'] = 1

        # Short Position Exit
        dataframe.loc[
            (
                    qtpylib.crossed_above(dataframe['moon_phase'],99) # exit short just before full moon
            ),
            'exit_short'] = 1

        return dataframe
