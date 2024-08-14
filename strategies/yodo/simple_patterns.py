# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, merge_informative_pair, informative)
from technical.util import resample_to_interval, resampled_merge

from freqtrade.persistence import Trade
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.pivots_points import pivots_points
from typing import Any, Dict, List, Optional

# 13% APR 1 year backtest
class simple_patterns(IStrategy):
    custom_info = {}
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 1

    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.03

    # Trailing stoploss
    trailing_stop= True
    trailing_stop_positive=0.02
    trailing_stop_positive_offset= 0.04
    trailing_only_offset_is_reached= False
    
    custom_price_max_distance_ratio = 1

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
        
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50


    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {

            },
            'subplots': {
                "MACD": {
                    'macdh': {'color': 'blue'},
                    'macdd': {'color': 'cyan'},
                    'macdf': {'color': 'purple'},
                },
                "CCI": {
                    'cci': {'color': 'red'},

                },
            }
        } 

    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }
        ]       
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:


        return (50)
    


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs
    
    @informative('3m')
    @informative('5m')
    @informative('15m')
    @informative('30m')
    @informative('1h')
    @informative('4h')
    @informative('12h')
    @informative('1d')
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['morning'] = ta.CDLMORNINGSTAR(dataframe) 
        dataframe['dragonfly'] = ta.CDLDRAGONFLYDOJI(dataframe)
        dataframe['hammer'] = ta.CDLHAMMER(dataframe)
        dataframe['harami'] = ta.CDLHARAMI(dataframe)
        dataframe['soldiers'] = ta.CDL3WHITESOLDIERS(dataframe)

        return dataframe



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MORNING #
        dataframe.loc[
            (
                (dataframe['morning_5m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'morning_5m')
        dataframe.loc[
            (
                (dataframe['morning_15m'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'morning_15m')
        dataframe.loc[
            (
                (dataframe['morning_30m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'morning_30m')
        dataframe.loc[
            (
                (dataframe['morning_1h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'morning_1h')
        dataframe.loc[
            (
                (dataframe['morning_4h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'morning_4h')
        # DRAGON #
        dataframe.loc[
            (
                (dataframe['dragonfly_5m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'dragonfly_5m')
        dataframe.loc[
            (
                (dataframe['dragonfly_15m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'dragonfly_15m')
        dataframe.loc[
            (
                (dataframe['dragonfly_30m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'dragonfly_30m')
        dataframe.loc[
            (
                (dataframe['dragonfly_1h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'dragonfly_1h')
        dataframe.loc[
            (
                (dataframe['dragonfly_4h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'dragonfly_4h')

        # HAMMER #
        dataframe.loc[
            (
                (dataframe['hammer_5m'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'hammer_5m')
        dataframe.loc[
            (
                (dataframe['hammer_15m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'hammer_15m')
        dataframe.loc[
            (
                (dataframe['hammer_30m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'hammer_30m')
        dataframe.loc[
            (
                (dataframe['hammer_1h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'hammer_1h')
        dataframe.loc[
            (
                (dataframe['hammer_1h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'hammer_1h')
        
        # HARAMI #
        dataframe.loc[
            (
                (dataframe['harami_5m'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'harami_5m')
        dataframe.loc[
            (
                (dataframe['harami_15m'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'harami_15m')
        dataframe.loc[
            (
                (dataframe['harami_30m'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'harami_30m')
        dataframe.loc[
            (
                (dataframe['harami_1h'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'harami_1h')
        dataframe.loc[
            (
                (dataframe['harami_4h'] != 0) &
                (dataframe['volume'] > 0) 
            ),
            ['enter_long', 'enter_tag']] = (1, 'harami_4h')
        
        # SOLDIERS #
        dataframe.loc[
            (
                (dataframe['soldiers_5m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'soldiers_5m')
        dataframe.loc[
            (
                (dataframe['soldiers_15m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'soldiers_15m')
        dataframe.loc[
            (
                (dataframe['soldiers_30m'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'soldiers_30m')
        dataframe.loc[
            (
                (dataframe['soldiers_1h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'soldiers_1h')
        dataframe.loc[
            (
                (dataframe['soldiers_4h'] != 0) &
                (dataframe['volume'] > 0)  
            ),
            ['enter_long', 'enter_tag']] = (1, 'soldiers_4h')
        
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (            
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (0, "placeholder")
        return dataframe
