# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import numpy as np
from technical.util import resample_to_interval, resampled_merge
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

# --------------------------------



class slownsteady(IStrategy):


    minimal_roi = {
         "0": 0.025,
         "6": 0.02,
         "14": 0.015,
         "32": 0.01,
         "60": 0.008,
         "120": 0.005
    }

    # Stoploss:
    stoploss = -0.25


    bb_spread_ma_value =  0.015 # 0.01978 #
    resampled_bb_spread_ma_value = 0.025 # 0.02497 #
    rsi_value = 30
    resampled_rsi_value =  40


    # Optimal timeframe for the strategy
    timeframe = '5m'

    timescale = 12# 12 minute
    timescale_large = 4 # 1 hr


    custom_info = {}

    order_types = {
        "buy": "limit",
        "sell": "market",
        "emergencysell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 600,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }



     # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True


    # ,

#
    plot_config = {
        'main_plot': {
             'resample_12_bbg_lowerband': {'color':'blue'},
            'resample_12_bbg_upperband': {'color': 'purple'},
        },
        'subplots': {
            "RSI":{
                'rsi':{'color':'green'}
            },
            "BB_spread":{
                'bb_spread_ma':{'color':'brown'}
            }
        }
    }

    startup_candle_count: int = 100

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe_macro = resample_to_interval(dataframe, self.get_ticker_indicator()*self.timescale)
        dataframe_macro['ATR'] = ta.ATR(dataframe_macro, timeperiod=14)

        dataframe_macro['rsi'] = ta.RSI(dataframe_macro, timeperiod=14)
        # dataframe_macro['ema21'] = ta.EMA(dataframe_macro, timeperiod=21)

        bollinger_macro = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe_macro), window=14, stds=2.5)
        dataframe_macro['bbg_lowerband'] = bollinger_macro['lower']
        dataframe_macro['bbg_middleband'] = bollinger_macro['mid']
        dataframe_macro['bbg_upperband'] = bollinger_macro['upper']

        dataframe_macro['bb_spread'] = (dataframe_macro['bbg_upperband'] - dataframe_macro['bbg_lowerband']) / dataframe_macro['bbg_middleband']
        dataframe_macro['bb_spread_ma'] = ta.SMA(dataframe_macro['bb_spread'],14)


        dataframe = resampled_merge(dataframe, dataframe_macro)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger_g = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=14, stds=2.5)
        dataframe['bbg_middleband'] = bollinger_g['mid']
        dataframe['bbg_upperband'] = bollinger_g['upper']
        dataframe['bbg_lowerband'] = bollinger_g['lower']


        dataframe['bb_spread'] = (dataframe['bbg_upperband'] - dataframe['bbg_lowerband']) / dataframe['bbg_middleband']
        dataframe['bb_spread_ma'] = ta.SMA(dataframe['bb_spread'],14)

        #
        dataframe['volume_rolling'] = dataframe['volume'].shift(14).rolling(14).mean()

        dataframe.fillna(method='ffill', inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_value) &
                (dataframe['close'] < dataframe['bbg_lowerband']) &
                (dataframe['close'].shift(1) < dataframe['bbg_lowerband'].shift(1)) &
                (dataframe['bb_spread_ma'] > self.bb_spread_ma_value) &
                (dataframe['bb_spread_ma'].shift(1) > self.bb_spread_ma_value).shift(1) &
                #(dataframe['resample_{}_bb_spread_ma'.format(self.get_ticker_indicator()*self.timescale)] > self.resampled_bb_spread_ma_value) &
                (dataframe['volume_rolling'] > 0) &
                (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*self.timescale)] < self.resampled_rsi_value ) &
                (dataframe['close'] < dataframe['resample_{}_bbg_lowerband'.format(self.get_ticker_indicator()*self.timescale)])
                #((dataframe['resample_60_crossed_above'].shift(1) == True) | (dataframe['resample_60_crossed_above'].shift(2)==True) | (dataframe['resample_60_crossed_above'] == True))
            ),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
