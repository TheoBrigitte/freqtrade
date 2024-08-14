# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter, informative, merge_informative_pair
# --------------------------------
# Add your lib to import here
from freqtrade.exchange import timeframe_to_minutes
# https://github.com/quantroom-pro/pyrenko
import pyrenko 
import pandas_ta as pta
from technical import qtpylib
import scipy.optimize as opt

logger = logging.getLogger(__name__)

def evaluate_renko(brick, history, column_name):
    renko_obj = pyrenko.renko()
    renko_obj.set_brick_size(brick_size = brick, auto = False)
    renko_obj.build_history(prices = history)
    return renko_obj.evaluate()[column_name]

class AdaptiveRenkoStrategy(IStrategy):
    
    """
    This strategy is based on research work by Sergey Malchevskiy and uses Renko bricks to identify trends.
    The brick size is optimized using ATR to find out boundaries from an informative period.
    For more information on the brick size optimization, see the following article: 
    https://towardsdatascience.com/renko-brick-size-optimization-34d64400f60e

    This strategy is designed to work with FreqTrade and can be used in conjunction with FreqAI.
    To match with research, brick size optimization should change from ATR to IQR from source data
    For more information on using Bayesian optimization on this strategy as a training model, see the following article: 
    https://towardsdatascience.com/bayesian-optimization-in-trading-77202ffed530

    Author: Mr Robot (@heymrrobot)
    """    
    
    INTERFACE_VERSION = 3

    timeframe = '3m'
    timeframe_minutes = timeframe_to_minutes(timeframe)
    can_short: bool = True
    trailing_stop = False
    startup_candle_count: int = 7*2
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True
    
    # Initialize renko_obj_obs as a dictionary
    custom_renkodict = {}
                       
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf_tf = '15m'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Calculate the ATR (Average True Range) for the informative DataFrame
        atr = pta.atr(informative['high'], informative['low'], informative['close'], timeperiod=10)
        # Add the ATR values as a new column to the informative DataFrame
        informative['atr'] = [0 if np.isnan(x) else x for x in atr]
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        
        # If the pair is not in the custom_renkodict dictionary, add it with a new pyrenko.renko() object
        if metadata['pair'] not in self.custom_renkodict:
            self.custom_renkodict[metadata['pair']] = pyrenko.renko()
            # Optimize the Renko brick size using fminbound for the new pyrenko.renko() object using the fminbound optimization function
            opt_bs = opt.fminbound(lambda x: -evaluate_renko(brick=x, history=dataframe.close, column_name='score'), np.min(dataframe['atr_15m']), np.max(dataframe['atr_15m']), disp=0)
            # Create a new pyrenko.renko() object for the pair with the optimized brick size and add it to the custom_renkodict dictionary
            self.custom_renkodict[metadata['pair']].set_brick_size(brick_size=opt_bs, auto=False)
            self.custom_renkodict[metadata['pair']].build_history(prices=dataframe.close)
            logger.info(f'Created a new pyrenko.renko() object for pair {metadata["pair"]}')
        
        # Get the renko object for the given pair from custom_renkodict    
        renko_obj = self.custom_renkodict[metadata['pair']]
        # Get the list of renko prices from the renko object
        prices = renko_obj.get_renko_prices()
        logger.info(f'Got the list of renko prices for pair {metadata["pair"]}')
        # Get the list of renko directions from the renko object
        directions = renko_obj.get_renko_directions()
        self.prev_brick_direction = directions[-2] if len(directions) >= 2 else None
        self.last_brick_direction = directions[-1] if len(directions) >= 1 else None
    
        # Check if there are any prices in the prices list
        if len(prices) > 0:
            # Create the next Renko bars using the last close price from the dataframe and get the number of created bars
            num_created_bars = renko_obj.do_next(dataframe.close.iloc[-1])
            logger.info(f'Created {num_created_bars} new Renko bars for pair {metadata["pair"]}')
            # If new bars were created
            if num_created_bars != 0:
                # If the previous brick direction is not the same as the last brick direction
                if self.prev_brick_direction != self.last_brick_direction:
                    opt_bs = opt.fminbound(lambda x: -evaluate_renko(brick=x, history=dataframe.close, column_name='score'), np.min(dataframe['atr_15m']), np.max(dataframe['atr_15m']), disp=0)
                    if opt_bs != renko_obj.brick_size:
                        renko_obj.set_brick_size(brick_size=opt_bs, auto=False)
                        self.opt_bs = opt_bs

        # Evaluate the Renko bars and update the score and directions dictionaries
        renko_eval = renko_obj.evaluate()
        logger.info(f'Evaluated the Renko bars for pair {metadata["pair"]}, values: {renko_eval}, brick size: {renko_obj.brick_size}')
        self.score = {metadata['pair']: renko_eval['score']}
        self.directions = {metadata['pair']: directions}
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add two new columns to the DataFrame for entering a long and short position, and initialize them to zero.
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Check if the current pair is not in the custom_renkodict, or if the score for the pair is less than 4, or if the length of the directions list for the pair is less than 2.
        # If any of these conditions is true, return the original DataFrame without making any changes.
        if metadata['pair'] not in self.custom_renkodict or self.score.get(metadata['pair'], -1) < 4 or len(self.directions[metadata['pair']]) < 2:
            return dataframe

        # Get the direction of the last brick and the direction of the second-to-last brick for the current pair from the directions list.
        last_brick_direction = self.directions[metadata['pair']][-1]
        prev_brick_direction = self.directions[metadata['pair']][-2]
        
        # If the direction of the last brick is up (1) and the direction of the previous brick is down (-1), 
        # set the value of enter_long to 1 and enter_tag to 'buy_direction' for all rows in the enter_long and enter_tag columns vice versa.
        if last_brick_direction == 1 and prev_brick_direction == -1:
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, 'buy_direction')
            logger.info(f'Set the value of enter_long to 1 and enter_tag to "buy_direction" for pair {metadata["pair"]}')
        elif last_brick_direction == -1 and prev_brick_direction == 1:
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, 'short_direction')
            logger.info(f'Set the value of enter_short to 1 and enter_tag to "short_direction" for pair {metadata["pair"]}')
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata['pair'] not in self.custom_renkodict:
            return dataframe
        
        last_brick_direction = self.directions[metadata['pair']][-1]
        prev_brick_direction = self.directions[metadata['pair']][-2]

        if last_brick_direction == -1 and prev_brick_direction == 1:
            dataframe.loc[:, ['exit_long', 'enter_tag']] = (1, 'direction_long_exit')
            logger.info(f'Set the value of exit_long to 1 and enter_tag to "direction_long_exit" for pair {metadata["pair"]}')
            # Flush the custom_renkodict for the corresponding pair
            # del self.custom_renkodict[metadata['pair']]

        elif last_brick_direction == 1 and prev_brick_direction == -1:
            dataframe.loc[:, ['exit_short', 'enter_tag']] = (1, 'direction_short_exit')
            logger.info(f'Set the value of exit_short to 1 and enter_tag to "direction_short_exit" for pair {metadata["pair"]}')
            # Flush the custom_renkodict for the corresponding pair
            # del self.custom_renkodict[metadata['pair']]
        return dataframe
