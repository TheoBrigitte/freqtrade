from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib
import numpy as np
import logging
import pandas as pd
import pandas_ta as pta
import datetime
from datetime import datetime, timedelta, timezone
from typing import Optional
import talib.abstract as ta
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from freqtrade.strategy import stoploss_from_open
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import technical.indicators as ftt

logger = logging.getLogger('freqtrade')


### Change log ###
# C.T. 3-9-23
# adding bull/bear detect of 1hr fast ewo
### Change log ###

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class twinturboV8(IStrategy):

    ### Strategy parameters ###
    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False # True
    ignore_roi_if_entry_signal = True
    use_exit_signal = True
    stoploss = -0.25
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    market_status = 0
    minimal_roi = {
        "0": 0.215,
        "40": 0.132,
        "87": 0.086,
        "201": 0.03
    }


    ### Hyperoptable parameters ###
    # entry optizimation
    max_epa = CategoricalParameter([2, 3], default=0, space="buy", optimize=True)

    # protections
    cooldown_lookback = IntParameter(24, 48, default=46, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 60, default=25, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 60, default=49, space='sell', optimize=True)
    low_offset = DecimalParameter(0.95, 0.99, default=0.97, decimals=2, space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=1.00,  decimals=2, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=1.3, decimals=2, space='sell', optimize=True)   

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = IntParameter(-5, 0, default=-4, space='buy', optimize=True)
    ewo_high = IntParameter(0, 2, default=0, space='buy', optimize=True)
    ewo_high_safe = IntParameter(2, 8, default=8, space='buy', optimize=True)
    ewo_high_uL = IntParameter(15, 25, default=20, space='buy', optimize=True)
    rsi_buy = IntParameter(55, 70, default=65, space='buy', optimize=True)
    ### BTC 
    bull = DecimalParameter(-0.25, 0.25, default=0, space='buy',decimals=2, optimize=True)
    estop = DecimalParameter(-0.5, 0, default=-0.5, space='sell',decimals=2, optimize=True)
    # dca level optimization
    dca1 = DecimalParameter(low=0.01, high=0.05, decimals=2, default=0.02, space='buy', optimize=True, load=True)
    dca2 = DecimalParameter(low=0.03, high=0.08, decimals=2, default=0.04, space='buy', optimize=True, load=True)
    dca3 = DecimalParameter(low=0.05, high=0.12, decimals=2, default=0.06, space='buy', optimize=True, load=True)

    #trailing stop loss optimiziation
    tsl_target5 = DecimalParameter(low=0.3, high=0.4, decimals=1, default=0.3, space='sell', optimize=True, load=True)
    ts5 = DecimalParameter(low=0.04, high=0.06, default=0.05, space='sell', optimize=True, load=True)
    tsl_target4 = DecimalParameter(low=0.18, high=0.3, default=0.2, space='sell', optimize=True, load=True)
    ts4 = DecimalParameter(low=0.03, high=0.05, default=0.045, space='sell', optimize=True, load=True)
    tsl_target3 = DecimalParameter(low=0.12, high=0.18, default=0.15, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.07, high=0.12, default=0.1, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.07, default=0.06, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.02, high=0.05, default=0.04, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.008, high=0.015, default=0.013, space='sell', optimize=True, load=True)

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 35
    
    ### protections ###
    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot


    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        pairs += ['BTC/USDT']
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    ### Dollar Cost Averaging ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier 

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2)

        if current_profit > -(self.dca1.value) and trade.nr_of_successful_entries == 1:
            return None

        if current_profit > -(self.dca2.value) and trade.nr_of_successful_entries == 2:
            return None

        if current_profit > -(self.dca3.value) and trade.nr_of_successful_entries == 3:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        # Grid Buy down every 2% with a small amount of capital
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if count_of_entries == 1: 
                stake_amount = stake_amount * 0.166
            elif count_of_entries == 2:
                stake_amount = stake_amount * 0.166
            elif count_of_entries == 3:
                stake_amount = stake_amount * 0.166
            else:
                stake_amount = stake_amount

            return stake_amount
        except Exception as exception:
            return None

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        for stop5 in self.tsl_target5.range:
            if (current_profit > stop5):
                for stop5a in self.ts5.range:
                    self.dp.send_msg(f'*** {pair} *** Profit: {current_profit} - lvl5 {stop5}/{stop5a} activated')
                    return stop5a 
        for stop4 in self.tsl_target4.range:
            if (current_profit > stop4):
                for stop4a in self.ts4.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl4 {stop4}/{stop4a} activated')
                    return stop4a 
        for stop3 in self.tsl_target3.range:
            if (current_profit > stop3):
                for stop3a in self.ts3.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {stop3}/{stop3a} activated')
                    return stop3a 
        for stop2 in self.tsl_target2.range:
            if (current_profit > stop2):
                for stop2a in self.ts2.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {stop2}/{stop2a} activated')
                    return stop2a 
        for stop1 in self.tsl_target1.range:
            if (current_profit > stop1):
                for stop1a in self.ts1.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {stop1}/{stop1a} activated')
                    return stop1a 
        for stop0 in self.tsl_target0.range:
            if (current_profit > stop0):
                for stop0a in self.ts0.range:
                    self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {stop0}/{stop0a} activated')
                    return stop0a 

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.dp:
            inf_tf = '1h'
            informative = self.dp.get_pair_dataframe(pair=f"BTC/USDT", timeframe=inf_tf)
        # BTC EWO 5/35
            informative['BTC_EWO_Fast'] = EWO(informative, 5, 35)

       ### Changed this part ###
        if qtpylib.crossed_above(informative['BTC_EWO_Fast'], self.bull.value).all():
            self.dp.send_msg(f"MARKET STATUS: Bear is gone! Lets F00kInG GOOOOO!!!", always_send=True)
            print("MARKET STATUS: Bear is gone! Lets F00kInG GOOOOO!!!")

        elif qtpylib.crossed_below(informative['BTC_EWO_Fast'], self.bull.value).all():
            self.dp.send_msg(f"MARKET STATUS: Bear Lurking! Grab the Lube, This could hurt...", always_send=True)
            print("MARKET STATUS: Bear Lurking! Grab the Lube, This could hurt...")

        elif qtpylib.crossed_below(informative['BTC_EWO_Fast'], self.estop.value).all():
            self.dp.send_msg(f"MARKET STATUS: ABANDON SHIP!!!", always_send=True)
            print("MARKET STATUS: ABANDON SHIP!!!")
        
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # HMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # SMA
        dataframe['200_SMA'] = ta.SMA(dataframe["close"], timeperiod = 200)

        # Plot 0
        dataframe['zero'] = 0

        # BTC SMA 5/30
 
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe['hma_50'] - dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_high_uL.value) &
                (dataframe['EWO'] > self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'].shift(1) > dataframe['EWO'].shift(2)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['rsi_fast'] < 35) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe < EWO2 & RSI Fast')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_high_uL.value) &
                (dataframe['EWO'] > self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'].shift(1) > dataframe['EWO'].shift(2)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe < EWO2')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_high_uL.value) &
                (dataframe['EWO'] > self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['rsi_fast'] < 35) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe < EWO1 & RSI Fast')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_high_uL.value) &
                (dataframe['EWO'] > self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe < EWO1')

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['close'] ,dataframe['hma_50'])) &
        #         (dataframe['close'] < (dataframe['hma_50'] * 1.0075))&
        #         (dataframe['EWO'] < 0.85) &
        #         (dataframe['EWO'] > self.estop.value) &
        #         (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']) &
        #         (dataframe['HMA_SQZ'] > -0.2) &
        #         (dataframe['rsi'] < self.rsi_buy.value) &
        #         (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'BTC Bull|HMA cross')

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['close'] ,dataframe['hma_50'])) &
        #         (dataframe['close'] < (dataframe['hma_50'] * 1.0075))&
        #         (dataframe['EWO'] < 0.85) &
        #         (dataframe['EWO'] > self.estop.value) &
        #         (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']) &
        #         (dataframe['hma_50'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']) &
        #         (dataframe['HMA_SQZ'] < 0.2) &
        #         (dataframe['rsi'] < self.rsi_buy.value) &
        #         (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
        #         (dataframe['volume'] > 0)
        #     ),
        #     ['enter_long', 'enter_tag']] = (1, 'BTC Bull|HMA cross|HMA SQZ')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['EWO'] < self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'].shift(1) > dataframe['EWO'].shift(2)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe > EWO2> High')


        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'].shift(1) > dataframe['EWO'].shift(2)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['rsi_fast'] < 35) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO2 < low & RSI Fast')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'].shift(1) > dataframe['EWO'].shift(2)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO2 < low')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['EWO'] < self.ewo_high_safe.value) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'Safe > EWO1 > High')


        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['rsi_fast'] < 35) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO1 < low & RSI Fast')

        dataframe.loc[
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > dataframe['EWO'].shift(1)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                # (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'EWO1 < low')

            # dataframe['BTC_EWO_Fast_1h'] > dataframe['BTC_EWO_Fast_1h'].shift.(1) & 
            #         dataframe['BTC_EWO_Fast_1h'].shift.(1) & dataframe['BTC_EWO_Fast_1h'].shift.(2))) & 

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ### possibly change sell signals?

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            ),
            ['exit_long', 'exit_tag']] = (1, 'Close > Offset Hi')

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            ),
            ['exit_long', 'exit_tag']] = (1, 'Close < Offset Lo')

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_below(dataframe['close'] ,dataframe['hma_50'])) &
        #         (dataframe['BTC_EWO_Fast_1h'] >= self.bull.value) &
        #         (dataframe['volume'] > 0)

        #     ),
        #     ['exit_long', 'exit_tag']] = (1, 'Close > Offset Lo')

        dataframe.loc[
            (
                (dataframe['BTC_EWO_Fast_1h'] <= self.estop.value) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'fucking bearzzz')

        return dataframe

# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): base_nb_candles_buy = 59
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): dca1 = 0.02
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): dca2 = 0.04
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): dca3 = 0.06
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ewo_high = 2.3
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ewo_high_safe = 2.3
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ewo_low = -9.0
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): fast_1h = 5
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): low_offset = 0.99
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_epa = 3
# 2023-02-28 18:02:48,341 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): rsi_buy = 68
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): slow_1h = 30
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): base_nb_candles_sell = 49
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): high_offset = 1.01
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): high_offset_2 = 1.02
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts0 = 0.013
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts1 = 0.013
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts2 = 0.02
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts3 = 0.035
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts4 = 0.045
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ts5 = 0.05
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target0 = 0.04
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target1 = 0.06
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target2 = 0.1
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target3 = 0.15
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target4 = 0.2
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): tsl_target5 = 0.3
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
# 2023-02-28 18:02:48,342 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): cooldown_lookback = 46
# 2023-02-28 18:02:48,343 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stop_duration = 5
# 2023-02-28 18:02:48,343 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): use_stop_protection = True
# 2023-02-28 18:02:48,370 - freqtrade.data.history.idatahandler - WARNING - BTC/USDT, spot, 1h, data ends at 2023-02-14 20:00:00
# 2023-02-28 18:02:56,776 - freqtrade.optimize.backtesting - INFO - Backtesting with data from 2023-01-01 00:00:00 up to 2023-02-25 00:00:00 (55 days).
##### INTIAL BACKTEST RESULTS #####
# Result for strategy twinturboV8
# ================================================================ BACKTESTING REPORT ===============================================================
# |            Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
# |-----------------+-----------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
# | HIVALHALLA/USDT |        15 |           4.72 |          70.81 |           209.314 |          20.93 |          0:55:00 |    14     0     1  93.3 |
# |        UBX/USDT |         5 |           8.10 |          40.48 |           168.750 |          16.88 |          0:38:00 |     5     0     0   100 |
# |     AGIX2S/USDT |         8 |           3.27 |          26.19 |           158.795 |          15.88 |          2:09:00 |     8     0     0   100 |
# |        KOL/USDT |         5 |           5.33 |          26.67 |           144.091 |          14.41 |          2:15:00 |     5     0     0   100 |
# |        SON/USDT |         7 |           4.23 |          29.60 |           133.183 |          13.32 |          2:56:00 |     6     0     1  85.7 |
# |    HIAZUKI/USDT |        14 |           2.45 |          34.33 |           128.458 |          12.85 |          2:22:00 |     8     0     6  57.1 |
# |       NDAU/USDT |        10 |           3.87 |          38.71 |           120.805 |          12.08 |          4:24:00 |    10     0     0   100 |
# |        TEM/USDT |        15 |           4.49 |          67.32 |           115.915 |          11.59 |          1:23:00 |    14     0     1  93.3 |
# |     VISION/USDT |         9 |           2.81 |          25.28 |           115.771 |          11.58 |          2:14:00 |     7     0     2  77.8 |
# |        GAS/USDT |         4 |           6.12 |          24.50 |           111.151 |          11.12 |          1:01:00 |     4     0     0   100 |
# |       SDAO/USDT |         7 |           2.81 |          19.65 |           108.673 |          10.87 |          0:54:00 |     6     0     1  85.7 |
# |       MITX/USDT |         8 |           3.68 |          29.40 |           108.564 |          10.86 |          0:51:00 |     7     0     1  87.5 |
# |        UPO/USDT |        10 |           3.35 |          33.55 |           102.870 |          10.29 |          1:14:00 |     9     0     1  90.0 |
# |      ERSDL/USDT |         5 |           3.74 |          18.72 |           102.712 |          10.27 |          3:49:00 |     5     0     0   100 |
# |       STND/USDT |         4 |           4.27 |          17.07 |           100.043 |          10.00 |          0:19:00 |     4     0     0   100 |
# |      GRT2L/USDT |         8 |           2.42 |          19.35 |            97.812 |           9.78 |          3:11:00 |     7     0     1  87.5 |
# |        FRM/USDT |         6 |           4.79 |          28.74 |            97.282 |           9.73 |          3:05:00 |     6     0     0   100 |
# |   HICLONEX/USDT |        10 |           2.46 |          24.58 |            96.439 |           9.64 |          0:52:00 |     7     0     3  70.0 |
# |       REVU/USDT |         8 |           3.44 |          27.50 |            94.302 |           9.43 |          2:09:00 |     7     0     1  87.5 |
# |       MNET/USDT |        17 |           2.66 |          45.17 |            93.026 |           9.30 |          1:55:00 |    12     0     5  70.6 |
# |       HAWK/USDT |        13 |           1.82 |          23.63 |            92.448 |           9.24 |          1:44:00 |     9     0     4  69.2 |
# |        ALT/USDT |         9 |           4.50 |          40.46 |            90.955 |           9.10 |          0:40:00 |     8     0     1  88.9 |
# |     AGIX2L/USDT |         4 |           3.33 |          13.34 |            86.773 |           8.68 |          1:40:00 |     4     0     0   100 |
# | HIPENGUINS/USDT |         9 |           2.31 |          20.82 |            86.053 |           8.61 |          1:54:00 |     7     0     2  77.8 |
# |       BEAT/USDT |         6 |           2.42 |          14.52 |            84.038 |           8.40 |          1:21:00 |     4     0     2  66.7 |
# |  STEPWATCH/USDT |        12 |           2.07 |          24.79 |            83.774 |           8.38 |          1:21:00 |     8     0     4  66.7 |
# |        RED/USDT |         7 |           2.46 |          17.24 |            83.116 |           8.31 |          1:26:00 |     4     0     3  57.1 |
# |       LOVE/USDT |         4 |           4.53 |          18.11 |            80.846 |           8.08 |          0:59:00 |     4     0     0   100 |
# |       2CRZ/USDT |         7 |           3.07 |          21.48 |            78.732 |           7.87 |          6:48:00 |     7     0     0   100 |
# |  HIFRIENDS/USDT |         5 |           2.76 |          13.79 |            77.395 |           7.74 |          0:23:00 |     4     0     1  80.0 |
# |        GGG/USDT |         4 |           2.35 |           9.39 |            76.417 |           7.64 |          2:06:00 |     3     0     1  75.0 |
# |        CTI/USDT |         8 |           2.87 |          22.94 |            75.736 |           7.57 |          1:23:00 |     7     0     1  87.5 |
# |        SPA/USDT |         3 |           4.16 |          12.47 |            72.060 |           7.21 |          2:52:00 |     3     0     0   100 |
# |        OLE/USDT |         6 |           2.43 |          14.56 |            70.431 |           7.04 |          2:52:00 |     4     0     2  66.7 |
# |     HIFLUF/USDT |         6 |           2.71 |          16.27 |            69.742 |           6.97 |          2:20:00 |     6     0     0   100 |
# |      DAPPX/USDT |         6 |           2.61 |          15.66 |            68.978 |           6.90 |          2:47:00 |     6     0     0   100 |
# |     SHIB2S/USDT |         4 |           3.30 |          13.22 |            68.252 |           6.83 |          7:46:00 |     4     0     0   100 |
# |       DSLA/USDT |         3 |           4.43 |          13.28 |            66.957 |           6.70 |          0:38:00 |     3     0     0   100 |
# |       KONO/USDT |         9 |           3.19 |          28.68 |            63.852 |           6.39 |          5:56:00 |     8     0     1  88.9 |
# |        MAN/USDT |         6 |           2.49 |          14.94 |            61.525 |           6.15 |          2:11:00 |     4     0     2  66.7 |
# |        ISP/USDT |         1 |          10.84 |          10.84 |            61.204 |           6.12 |          1:15:00 |     1     0     0   100 |
# |      SUTER/USDT |         3 |           3.95 |          11.84 |            60.269 |           6.03 |          0:42:00 |     3     0     0   100 |
# |      PUMLX/USDT |         3 |           4.49 |          13.48 |            60.119 |           6.01 |          0:32:00 |     3     0     0   100 |
# |       MONI/USDT |         6 |           2.29 |          13.75 |            59.982 |           6.00 |         12:32:00 |     5     0     1  83.3 |
# |        VXV/USDT |         8 |           1.71 |          13.64 |            55.067 |           5.51 |          3:45:00 |     7     0     1  87.5 |
# |       CSIX/USDT |         3 |           2.88 |           8.64 |            54.833 |           5.48 |          1:48:00 |     3     0     0   100 |
# |        FRA/USDT |         3 |           4.00 |          12.00 |            53.637 |           5.36 |          0:33:00 |     3     0     0   100 |
# |        PBX/USDT |         4 |           4.21 |          16.85 |            53.567 |           5.36 |          5:10:00 |     3     0     1  75.0 |
# |        OGV/USDT |         4 |           2.19 |           8.78 |            53.067 |           5.31 |          1:22:00 |     3     0     1  75.0 |
# |       MPLX/USDT |         6 |           3.03 |          18.21 |            50.647 |           5.06 |          2:13:00 |     6     0     0   100 |
# |      VAIOT/USDT |        12 |           1.70 |          20.44 |            50.546 |           5.05 |          4:33:00 |     9     0     3  75.0 |
# |     PRIMAL/USDT |         6 |           3.32 |          19.94 |            49.765 |           4.98 |          0:52:00 |     6     0     0   100 |
# |        KAR/USDT |         1 |          13.19 |          13.19 |            47.993 |           4.80 |          1:00:00 |     1     0     0   100 |
# |       DODO/USDT |         2 |           3.85 |           7.71 |            47.861 |           4.79 |          0:55:00 |     2     0     0   100 |
# |        CFX/USDT |         6 |           2.08 |          12.46 |            47.491 |           4.75 |          2:51:00 |     4     0     2  66.7 |
# |        CWS/USDT |         2 |           6.18 |          12.36 |            47.255 |           4.73 |          1:38:00 |     2     0     0   100 |
# |        EFX/USDT |         7 |           2.73 |          19.14 |            45.659 |           4.57 |          2:19:00 |     7     0     0   100 |
# |      ARKER/USDT |         4 |           2.88 |          11.53 |            43.916 |           4.39 |          7:39:00 |     2     0     2  50.0 |
# |       OP2S/USDT |         2 |           4.55 |           9.09 |            42.830 |           4.28 |          1:05:00 |     2     0     0   100 |
# |        BNC/USDT |         8 |           1.72 |          13.76 |            42.444 |           4.24 |          5:08:00 |     5     0     3  62.5 |
# |        RBP/USDT |         5 |           1.72 |           8.59 |            41.474 |           4.15 |          2:06:00 |     4     0     1  80.0 |
# |        AFK/USDT |         2 |           3.59 |           7.18 |            40.988 |           4.10 |          0:30:00 |     2     0     0   100 |
# |       PERP/USDT |         2 |           2.82 |           5.64 |            40.686 |           4.07 |          2:22:00 |     2     0     0   100 |
# |      QUICK/USDT |         1 |           6.25 |           6.25 |            40.641 |           4.06 |          0:35:00 |     1     0     0   100 |
# |        ARX/USDT |         4 |           5.68 |          22.73 |            40.639 |           4.06 |          1:31:00 |     4     0     0   100 |
# |      BEPRO/USDT |         6 |           1.93 |          11.59 |            39.775 |           3.98 |          3:07:00 |     5     0     1  83.3 |
# |        OVR/USDT |         1 |           8.59 |           8.59 |            39.402 |           3.94 |          2:55:00 |     1     0     0   100 |
# |       MAKI/USDT |         4 |           2.92 |          11.67 |            39.043 |           3.90 |          1:45:00 |     4     0     0   100 |
# |      APT2S/USDT |         2 |           3.41 |           6.82 |            38.785 |           3.88 |          3:58:00 |     2     0     0   100 |
# |       IDEA/USDT |         3 |           2.28 |           6.85 |            38.769 |           3.88 |          2:57:00 |     2     0     1  66.7 |
# |       WILD/USDT |         2 |           4.61 |           9.22 |            36.612 |           3.66 |          2:18:00 |     2     0     0   100 |
# |       BULL/USDT |         3 |           2.09 |           6.27 |            36.594 |           3.66 |          0:17:00 |     2     0     1  66.7 |
# |       EVER/USDT |         1 |           6.08 |           6.08 |            36.542 |           3.65 |          0:05:00 |     1     0     0   100 |
# |      HEART/USDT |         5 |           1.63 |           8.15 |            36.424 |           3.64 |          4:35:00 |     3     0     2  60.0 |
# |        AKT/USDT |         2 |           4.61 |           9.21 |            36.187 |           3.62 |          0:55:00 |     2     0     0   100 |
# |       OPUL/USDT |         1 |           7.42 |           7.42 |            35.358 |           3.54 |          0:30:00 |     1     0     0   100 |
# |        PKF/USDT |         6 |           1.40 |           8.37 |            35.302 |           3.53 |          7:20:00 |     4     0     2  66.7 |
# |        BFC/USDT |         5 |           2.88 |          14.41 |            35.011 |           3.50 |         11:34:00 |     5     0     0   100 |
# |        BBC/USDT |         7 |           2.61 |          18.24 |            34.296 |           3.43 |          1:49:00 |     6     0     1  85.7 |
# |       PHNX/USDT |         8 |           2.80 |          22.42 |            33.975 |           3.40 |          2:01:00 |     7     0     1  87.5 |
# |     BLUR2L/USDT |         3 |           1.56 |           4.67 |            33.963 |           3.40 |          2:17:00 |     2     0     1  66.7 |
# |       BOLT/USDT |         2 |           3.36 |           6.73 |            33.412 |           3.34 |          4:32:00 |     2     0     0   100 |
# |        ADX/USDT |         2 |           6.49 |          12.99 |            33.054 |           3.31 |          2:20:00 |     2     0     0   100 |
# |    SWINGBY/USDT |         3 |           2.84 |           8.51 |            32.971 |           3.30 |          3:53:00 |     3     0     0   100 |
# |        SDL/USDT |         5 |           1.17 |           5.83 |            32.865 |           3.29 |          0:10:00 |     2     0     3  40.0 |
# |        ELA/USDT |         2 |           3.13 |           6.26 |            32.196 |           3.22 |          0:08:00 |     2     0     0   100 |
# |     DREAMS/USDT |         3 |           2.27 |           6.80 |            31.636 |           3.16 |          5:38:00 |     2     0     1  66.7 |
# |        CMP/USDT |         2 |           4.33 |           8.66 |            30.948 |           3.09 |          2:10:00 |     2     0     0   100 |
# |        PRE/USDT |         1 |           5.99 |           5.99 |            30.476 |           3.05 |          1:25:00 |     1     0     0   100 |
# |     PSTAKE/USDT |         4 |           1.47 |           5.90 |            30.137 |           3.01 |          3:58:00 |     3     0     1  75.0 |
# |        PEL/USDT |         3 |           2.63 |           7.89 |            29.756 |           2.98 |          3:48:00 |     3     0     0   100 |
# |        OAS/USDT |         3 |           3.06 |           9.19 |            29.535 |           2.95 |          2:42:00 |     3     0     0   100 |
# |       OP2L/USDT |         2 |           2.39 |           4.77 |            29.459 |           2.95 |          0:28:00 |     2     0     0   100 |
# |        TVK/USDT |         2 |           3.47 |           6.94 |            29.027 |           2.90 |          0:48:00 |     2     0     0   100 |
# |      EGAME/USDT |         3 |           2.04 |           6.12 |            28.898 |           2.89 |          0:30:00 |     2     0     1  66.7 |
# |       WELL/USDT |         2 |           1.95 |           3.91 |            28.595 |           2.86 |          4:22:00 |     1     0     1  50.0 |
# |      BRISE/USDT |         3 |           2.84 |           8.51 |            28.473 |           2.85 |          1:45:00 |     3     0     0   100 |
# |     QUARTZ/USDT |         4 |           1.11 |           4.43 |            28.401 |           2.84 |          1:38:00 |     2     0     2  50.0 |
# |    HISEALS/USDT |         1 |           4.51 |           4.51 |            28.087 |           2.81 |          0:05:00 |     1     0     0   100 |
# |       CRPT/USDT |         6 |           2.84 |          17.06 |            27.939 |           2.79 |          0:22:00 |     5     0     1  83.3 |
# |       SCLP/USDT |         3 |           3.32 |           9.96 |            27.777 |           2.78 |          3:27:00 |     3     0     0   100 |
# |        IOI/USDT |         3 |           5.74 |          17.22 |            27.690 |           2.77 |          2:37:00 |     3     0     0   100 |
# |        LTO/USDT |         1 |           3.95 |           3.95 |            27.169 |           2.72 |          3:05:00 |     1     0     0   100 |
# |        PRQ/USDT |         1 |           3.37 |           3.37 |            26.823 |           2.68 |          2:30:00 |     1     0     0   100 |
# |       RMRK/USDT |         4 |           2.29 |           9.15 |            26.487 |           2.65 |          4:34:00 |     3     0     1  75.0 |
# |      MARSH/USDT |         2 |           2.30 |           4.59 |            25.871 |           2.59 |          2:20:00 |     2     0     0   100 |
# |        HBB/USDT |         1 |           3.00 |           3.00 |            25.507 |           2.55 |          8:00:00 |     1     0     0   100 |
# |     SHIB2L/USDT |         3 |           1.97 |           5.92 |            25.317 |           2.53 |          3:33:00 |     2     0     1  66.7 |
# |        IGU/USDT |         1 |           4.23 |           4.23 |            24.880 |           2.49 |          1:20:00 |     1     0     0   100 |
# |        STX/USDT |         2 |           2.91 |           5.82 |            24.712 |           2.47 |          7:45:00 |     2     0     0   100 |
# |       POKT/USDT |         1 |           5.60 |           5.60 |            24.466 |           2.45 |          1:15:00 |     1     0     0   100 |
# |      COOHA/USDT |        11 |           0.16 |           1.77 |            24.304 |           2.43 |          3:09:00 |     8     0     3  72.7 |
# |        FXS/USDT |         1 |           3.42 |           3.42 |            23.425 |           2.34 |          1:35:00 |     1     0     0   100 |
# |      FLAME/USDT |         4 |           1.78 |           7.10 |            23.424 |           2.34 |          1:46:00 |     2     0     2  50.0 |
# |        LIT/USDT |         1 |           3.00 |           3.00 |            22.815 |           2.28 |          7:55:00 |     1     0     0   100 |
# |       LABS/USDT |         5 |           1.62 |           8.08 |            22.639 |           2.26 |          2:14:00 |     4     0     1  80.0 |
# |       DEGO/USDT |         1 |           4.44 |           4.44 |            22.130 |           2.21 |          2:15:00 |     1     0     0   100 |
# |        VID/USDT |         1 |           8.02 |           8.02 |            22.073 |           2.21 |          1:15:00 |     1     0     0   100 |
# |       REVV/USDT |         1 |           3.75 |           3.75 |            21.868 |           2.19 |          0:25:00 |     1     0     0   100 |
# |       ROSN/USDT |         6 |           0.75 |           4.52 |            21.581 |           2.16 |          0:58:00 |     3     0     3  50.0 |
# |        COV/USDT |         5 |           2.12 |          10.59 |            21.404 |           2.14 |          2:03:00 |     4     0     1  80.0 |
# |        UNO/USDT |         1 |           4.20 |           4.20 |            20.912 |           2.09 |          3:25:00 |     1     0     0   100 |
# |   HIUNDEAD/USDT |         1 |           2.91 |           2.91 |            20.866 |           2.09 |         12:35:00 |     1     0     0   100 |
# |       NHCT/USDT |         2 |           1.91 |           3.81 |            20.174 |           2.02 |          2:20:00 |     2     0     0   100 |
# |      DAPPT/USDT |         2 |           3.48 |           6.96 |            20.166 |           2.02 |          2:02:00 |     1     0     1  50.0 |
# |        CKB/USDT |         1 |           2.98 |           2.98 |            20.164 |           2.02 |          1:35:00 |     1     0     0   100 |
# |        XNL/USDT |         2 |           3.00 |           5.99 |            19.743 |           1.97 |          5:38:00 |     2     0     0   100 |
# |       QRDO/USDT |         3 |           3.26 |           9.79 |            19.577 |           1.96 |          1:37:00 |     3     0     0   100 |
# |        ACS/USDT |         1 |           3.13 |           3.13 |            19.496 |           1.95 |          0:20:00 |     1     0     0   100 |
# |        PBR/USDT |         1 |           4.17 |           4.17 |            19.403 |           1.94 |          0:35:00 |     1     0     0   100 |
# |       PLGR/USDT |         3 |           1.55 |           4.66 |            19.009 |           1.90 |          3:08:00 |     2     0     1  66.7 |
# |       VIDT/USDT |         2 |           2.51 |           5.02 |            18.855 |           1.89 |          1:18:00 |     2     0     0   100 |
# |        SHX/USDT |         2 |           4.07 |           8.13 |            18.684 |           1.87 |          0:18:00 |     2     0     0   100 |
# |       AGIX/USDT |         1 |           3.23 |           3.23 |            18.672 |           1.87 |          0:25:00 |     1     0     0   100 |
# |      P00LS/USDT |         1 |           5.36 |           5.36 |            18.557 |           1.86 |          0:35:00 |     1     0     0   100 |
# |       MOOV/USDT |         3 |           2.91 |           8.72 |            18.551 |           1.86 |          2:32:00 |     3     0     0   100 |
# |       UNIC/USDT |         1 |           3.00 |           3.00 |            18.533 |           1.85 |          6:30:00 |     1     0     0   100 |
# |        TRU/USDT |         2 |           1.08 |           2.17 |            17.981 |           1.80 |          2:10:00 |     1     0     1  50.0 |
# |        GMB/USDT |         4 |           2.63 |          10.53 |            17.152 |           1.72 |          2:15:00 |     4     0     0   100 |
# |        ACQ/USDT |         2 |           4.28 |           8.56 |            17.133 |           1.71 |          1:15:00 |     2     0     0   100 |
# |      NTVRK/USDT |         3 |           2.78 |           8.33 |            16.943 |           1.69 |          4:00:00 |     2     0     1  66.7 |
# |      TRIAS/USDT |         1 |           3.00 |           3.00 |            16.744 |           1.67 |          3:25:00 |     1     0     0   100 |
# |       AKRO/USDT |         1 |           2.95 |           2.95 |            16.399 |           1.64 |          1:05:00 |     1     0     0   100 |
# |       PDEX/USDT |         1 |           3.32 |           3.32 |            16.060 |           1.61 |          0:10:00 |     1     0     0   100 |
# |         FT/USDT |         1 |           3.27 |           3.27 |            16.051 |           1.61 |          1:15:00 |     1     0     0   100 |
# |        YLD/USDT |         1 |           2.41 |           2.41 |            16.044 |           1.60 |          6:55:00 |     1     0     0   100 |
# |       DVPN/USDT |         2 |           1.44 |           2.87 |            15.787 |           1.58 |          9:55:00 |     1     0     1  50.0 |
# |      SENSO/USDT |         1 |           3.00 |           3.00 |            15.495 |           1.55 |          3:45:00 |     1     0     0   100 |
# |    HIRENGA/USDT |         1 |           3.86 |           3.86 |            15.469 |           1.55 |          0:30:00 |     1     0     0   100 |
# |       WOOP/USDT |         3 |           2.98 |           8.94 |            15.442 |           1.54 |          9:17:00 |     3     0     0   100 |
# |      APT2L/USDT |         1 |           2.95 |           2.95 |            15.400 |           1.54 |          1:20:00 |     1     0     0   100 |
# |        SXP/USDT |         1 |           2.71 |           2.71 |            15.338 |           1.53 |          3:25:00 |     1     0     0   100 |
# |       GENS/USDT |         5 |           1.91 |           9.57 |            15.269 |           1.53 |          2:38:00 |     3     0     2  60.0 |
# |        TXA/USDT |         3 |           2.35 |           7.06 |            15.261 |           1.53 |          3:23:00 |     3     0     0   100 |
# |     RACEFI/USDT |         7 |           1.78 |          12.44 |            14.894 |           1.49 |          3:54:00 |     6     0     1  85.7 |
# |       SHFT/USDT |         4 |           2.09 |           8.34 |            14.695 |           1.47 |          3:55:00 |     4     0     0   100 |
# |       NRFB/USDT |         3 |           2.44 |           7.31 |            14.347 |           1.43 |          2:10:00 |     3     0     0   100 |
# |        CLH/USDT |         5 |           1.12 |           5.59 |            13.372 |           1.34 |          1:58:00 |     3     0     2  60.0 |
# |       POLK/USDT |         2 |           2.09 |           4.17 |            13.140 |           1.31 |          7:00:00 |     2     0     0   100 |
# |        SHR/USDT |         2 |           3.54 |           7.08 |            12.910 |           1.29 |          1:10:00 |     2     0     0   100 |
# |        KYL/USDT |         2 |          -1.79 |          -3.57 |            12.843 |           1.28 |   1 day, 2:10:00 |     1     0     1  50.0 |
# | HICOOLCATS/USDT |         4 |           0.62 |           2.47 |            12.749 |           1.27 |          8:05:00 |     3     0     1  75.0 |
# |       GARI/USDT |         3 |           1.90 |           5.69 |            12.050 |           1.21 |          3:00:00 |     2     0     1  66.7 |
# |        DYP/USDT |         2 |           2.80 |           5.60 |            11.987 |           1.20 |          3:55:00 |     2     0     0   100 |
# |        XYO/USDT |         3 |           3.34 |          10.01 |            11.795 |           1.18 |          0:50:00 |     3     0     0   100 |
# |      ACOIN/USDT |         2 |           0.96 |           1.92 |            11.400 |           1.14 |          7:32:00 |     1     0     1  50.0 |
# |       ORBS/USDT |         1 |           3.00 |           3.00 |            11.382 |           1.14 |          3:35:00 |     1     0     0   100 |
# |        NYM/USDT |         1 |           3.00 |           3.00 |            11.096 |           1.11 |          6:10:00 |     1     0     0   100 |
# |      VOXEL/USDT |         1 |           3.06 |           3.06 |            11.091 |           1.11 |          1:50:00 |     1     0     0   100 |
# |       DFYN/USDT |         2 |           3.30 |           6.59 |            10.936 |           1.09 |          7:00:00 |     2     0     0   100 |
# |       XTAG/USDT |         2 |           3.18 |           6.36 |            10.615 |           1.06 |          0:28:00 |     2     0     0   100 |
# |        FTM/USDT |         2 |           3.14 |           6.27 |            10.596 |           1.06 |          3:38:00 |     2     0     0   100 |
# |       MTRG/USDT |         1 |           3.00 |           3.00 |            10.361 |           1.04 |          3:30:00 |     1     0     0   100 |
# |       RFOX/USDT |         1 |           3.00 |           3.00 |             9.218 |           0.92 |         10:05:00 |     1     0     0   100 |
# |      GALAX/USDT |         2 |           3.35 |           6.70 |             9.105 |           0.91 |          2:22:00 |     2     0     0   100 |
# |        FLY/USDT |         2 |           3.50 |           7.01 |             9.061 |           0.91 |          0:52:00 |     2     0     0   100 |
# |       GODS/USDT |         1 |           3.00 |           3.00 |             8.516 |           0.85 |          6:00:00 |     1     0     0   100 |
# |      LAYER/USDT |         1 |           2.70 |           2.70 |             8.514 |           0.85 |          5:15:00 |     1     0     0   100 |
# |    FALCONS/USDT |         1 |           2.03 |           2.03 |             8.411 |           0.84 |          0:15:00 |     1     0     0   100 |
# |       ETHW/USDT |         2 |           3.00 |           5.99 |             8.064 |           0.81 |          6:38:00 |     2     0     0   100 |
# |        SHA/USDT |         1 |           3.00 |           3.00 |             7.895 |           0.79 |          3:00:00 |     1     0     0   100 |
# |          R/USDT |         1 |           3.27 |           3.27 |             7.719 |           0.77 |          1:25:00 |     1     0     0   100 |
# |        LMR/USDT |         1 |           2.80 |           2.80 |             7.652 |           0.77 |          1:35:00 |     1     0     0   100 |
# |        SYS/USDT |         2 |           4.58 |           9.17 |             7.572 |           0.76 |          0:25:00 |     1     0     1  50.0 |
# |        APT/USDT |         2 |           3.14 |           6.28 |             7.514 |           0.75 |          1:25:00 |     2     0     0   100 |
# |         AR/USDT |         1 |           3.60 |           3.60 |             7.513 |           0.75 |          1:50:00 |     1     0     0   100 |
# |        ENQ/USDT |         1 |           3.14 |           3.14 |             7.512 |           0.75 |          1:30:00 |     1     0     0   100 |
# |          T/USDT |         1 |           2.96 |           2.96 |             7.499 |           0.75 |          2:25:00 |     1     0     0   100 |
# |       DYDX/USDT |         2 |           1.52 |           3.04 |             7.407 |           0.74 |         11:30:00 |     2     0     0   100 |
# |       UNFI/USDT |         1 |           3.00 |           3.00 |             7.360 |           0.74 |          6:20:00 |     1     0     0   100 |
# |       NAKA/USDT |         1 |           3.03 |           3.03 |             7.342 |           0.73 |          0:20:00 |     1     0     0   100 |
# |      MAGIC/USDT |         1 |           2.89 |           2.89 |             7.283 |           0.73 |          0:10:00 |     1     0     0   100 |
# |        FSN/USDT |         1 |           3.75 |           3.75 |             7.278 |           0.73 |          0:25:00 |     1     0     0   100 |
# |        LSS/USDT |         1 |           1.17 |           1.17 |             7.241 |           0.72 |         15:20:00 |     1     0     0   100 |
# |       NEER/USDT |         1 |           3.00 |           3.00 |             7.194 |           0.72 |         13:05:00 |     1     0     0   100 |
# |      LPOOL/USDT |         1 |           3.21 |           3.21 |             6.930 |           0.69 |          1:10:00 |     1     0     0   100 |
# |        JAM/USDT |         4 |           0.82 |           3.28 |             6.267 |           0.63 |          2:44:00 |     2     0     2  50.0 |
# |       GOM2/USDT |         2 |           1.62 |           3.24 |             6.228 |           0.62 |          9:42:00 |     2     0     0   100 |
# |       COTI/USDT |         1 |           3.00 |           3.00 |             5.992 |           0.60 |          6:35:00 |     1     0     0   100 |
# |        SKU/USDT |         1 |           3.00 |           3.00 |             5.944 |           0.59 |          3:30:00 |     1     0     0   100 |
# |       EOSC/USDT |         7 |           0.39 |           2.76 |             5.900 |           0.59 |          1:34:00 |     3     0     4  42.9 |
# |        XHV/USDT |         1 |           3.70 |           3.70 |             5.799 |           0.58 |          2:55:00 |     1     0     0   100 |
# |        NGC/USDT |         1 |           3.00 |           3.00 |             5.582 |           0.56 |   1 day, 6:05:00 |     1     0     0   100 |
# |        EQX/USDT |         2 |           0.67 |           1.35 |             5.538 |           0.55 |          0:12:00 |     1     0     1  50.0 |
# |        GEM/USDT |         2 |           1.35 |           2.70 |             5.419 |           0.54 |          0:02:00 |     1     0     1  50.0 |
# |      HEGIC/USDT |         1 |           3.82 |           3.82 |             5.268 |           0.53 |          1:05:00 |     1     0     0   100 |
# |        VRA/USDT |         2 |           1.41 |           2.82 |             5.202 |           0.52 |          0:12:00 |     1     0     1  50.0 |
# |       DIVI/USDT |         1 |           3.68 |           3.68 |             5.114 |           0.51 |          0:05:00 |     1     0     0   100 |
# |        MNW/USDT |         1 |           2.14 |           2.14 |             4.949 |           0.49 |          6:40:00 |     1     0     0   100 |
# |        XNO/USDT |         1 |           2.00 |           2.00 |             4.873 |           0.49 |  2 days, 5:15:00 |     1     0     0   100 |
# |       REAP/USDT |         1 |           3.00 |           3.00 |             4.818 |           0.48 |          5:50:00 |     1     0     0   100 |
# |      LOOKS/USDT |         1 |           3.13 |           3.13 |             4.816 |           0.48 |          2:20:00 |     1     0     0   100 |
# |       LYXE/USDT |         1 |           3.69 |           3.69 |             4.695 |           0.47 |          3:25:00 |     1     0     0   100 |
# |       OSMO/USDT |         1 |           3.00 |           3.00 |             4.691 |           0.47 |          8:40:00 |     1     0     0   100 |
# |        PLY/USDT |         3 |           0.59 |           1.76 |             4.667 |           0.47 |          6:07:00 |     2     0     1  66.7 |
# |        CGG/USDT |         2 |           0.48 |           0.96 |             4.357 |           0.44 |          2:10:00 |     1     0     1  50.0 |
# |      MARS4/USDT |         1 |           3.07 |           3.07 |             4.308 |           0.43 |          1:45:00 |     1     0     0   100 |
# |        GST/USDT |         1 |           3.37 |           3.37 |             4.240 |           0.42 |          2:25:00 |     1     0     0   100 |
# |       EPIK/USDT |         1 |           3.40 |           3.40 |             4.208 |           0.42 |          0:30:00 |     1     0     0   100 |
# |       SUKU/USDT |         4 |           0.46 |           1.83 |             4.148 |           0.41 |          5:41:00 |     3     0     1  75.0 |
# |       GEEQ/USDT |         2 |           0.72 |           1.43 |             4.107 |           0.41 |          8:42:00 |     1     0     1  50.0 |
# |        ACT/USDT |         1 |           3.30 |           3.30 |             3.922 |           0.39 |          1:30:00 |     1     0     0   100 |
# |       MAHA/USDT |         1 |           2.62 |           2.62 |             3.886 |           0.39 |          1:30:00 |     1     0     0   100 |
# |      BIFIF/USDT |         2 |           1.56 |           3.12 |             3.865 |           0.39 |          1:30:00 |     1     0     1  50.0 |
# |       AUSD/USDT |         5 |           1.39 |           6.95 |             3.666 |           0.37 |          2:45:00 |     2     0     3  40.0 |
# |        EUL/USDT |         1 |           3.32 |           3.32 |             3.643 |           0.36 |          0:25:00 |     1     0     0   100 |
# |     MATTER/USDT |         2 |           0.72 |           1.45 |             3.538 |           0.35 |          2:55:00 |     1     0     1  50.0 |
# |       MNST/USDT |         1 |           2.41 |           2.41 |             3.261 |           0.33 |          0:25:00 |     1     0     0   100 |
# |     AURORA/USDT |         2 |           1.25 |           2.51 |             2.557 |           0.26 |         10:12:00 |     1     0     1  50.0 |
# |       DERC/USDT |         1 |           1.52 |           1.52 |             2.487 |           0.25 |          8:00:00 |     1     0     0   100 |
# |     STRONG/USDT |         1 |           1.70 |           1.70 |             2.335 |           0.23 |          0:15:00 |     1     0     0   100 |
# |       LINA/USDT |         2 |           0.08 |           0.17 |             2.001 |           0.20 |          5:18:00 |     1     0     1  50.0 |
# |      TIDAL/USDT |         1 |           0.30 |           0.30 |             1.658 |           0.17 |  1 day, 13:35:00 |     1     0     0   100 |
# |         DG/USDT |         1 |           0.44 |           0.44 |             1.605 |           0.16 |         19:35:00 |     1     0     0   100 |
# |       LITH/USDT |         1 |           0.37 |           0.37 |             1.519 |           0.15 |         10:45:00 |     1     0     0   100 |
# | FORESTPLUS/USDT |         3 |           0.96 |           2.89 |             1.193 |           0.12 |          5:02:00 |     1     0     2  33.3 |
# |        AVA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MTV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        KMD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TEL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         TT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      AERGO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XMR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ATOM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ETN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CHR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BNB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ALGO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XEM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XTZ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ZEC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ADA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WXT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ARPA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CHZ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       NOIA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WIN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DERO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BTT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ONE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       TOKO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MAP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DAG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        POL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NWC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        KSM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DASH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XDB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WOM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DGB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       COMP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CRO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        KAI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ORN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BNS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MKR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MLK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        JST/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DIA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       LINK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DOT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        EWT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        UMA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SUN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BUY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        YFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       LOKI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        UNI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        UOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NIM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      RFUEL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        FIL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       AAVE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BSV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XLM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ETC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ETH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        AOA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SNX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LYM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XRP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TRX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ONT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        VET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NEO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        EOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BCH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       GRIN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ROSE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PLU/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GRT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      MSWAP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      1INCH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LOC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        HTR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      FRONT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      HYDRA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CRV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SUSHI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ZEN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        REN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LRC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BOA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      THETA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        QNT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BAT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DOGE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DAO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CAKE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ZEE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LTX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PHA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SRK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       AVAX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       MANA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       RNDR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        RLY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       TARA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       IOST/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PCX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ANKR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SAND/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       FLUX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ZIL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      BOSON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     PUNDIX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       WAXP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         HT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        HAI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      FORTH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GHX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      TOWER/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CARD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XDC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SHIB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        KDA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ICP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        STC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       GOVI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        FKX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CELO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      MATIC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        OGN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       OUSD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MXC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HYVE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PYR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       PROM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       XCAD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ELON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       POLS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ABBC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       NORD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       GMEE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SFUND/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       XAVA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NFT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       AIOZ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HAPI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     MODEFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      YFDAI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       FORM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ARRR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NGM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LPT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ASD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       BOND/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SOUL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       NEAR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        OOE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CFG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        AXS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CLV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      ROUTE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       PMON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DPET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ERG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SOL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SLP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XCH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MTL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       IOTX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        REQ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      CIRUS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         QI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PNT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XPR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      TRIBE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       MOVR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WOO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        OXT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BAL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      STORJ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        YGG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SKL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TRB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        RLC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       XPRT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       EGLD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HBAR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       FLOW/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NKN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MLN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SOLVE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CTSI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      ALICE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ILV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SLIM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TLM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DEXE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       RUNE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        C98/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       PUSH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       AGLD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       TONE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       REEF/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        INJ/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      ALPHA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      JASMY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SUPER/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HERO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XED/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       AURY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SWASH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BUX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WRX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CERE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SHILL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ERN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       PAXG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      AUDIO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ENS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XTM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ATA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ANT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TWT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         OM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GLM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        VLX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       LIKE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       KAVA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     BURGER/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      CREAM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        RSR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        IMX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HARD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       POND/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NGL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       KDON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       TRVL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     BONDLY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XEC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       GAFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        UFO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ORC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     PEOPLE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      OCEAN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      WHALE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CWEB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       IOTA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        HNT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       GLMR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ASTR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        CVX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        AMP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MJT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     STARLY/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     ONSTON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WMT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      TFUEL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      METIS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      LAVAX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        WAL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      MELOS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        APE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       BICO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        STG/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       LOKA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       URUS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      TITAN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CEEK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     ALPINE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SYNR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DAR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         MV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      XDEFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       RACA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SWFTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        KNC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PLD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        EPX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BSW/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      FITFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GMM/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SIN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        DFA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        MBL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CELT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       DUSK/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       USDD/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SCRT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      STORE/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       LUNC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       USTC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         OP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        IHC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        ICX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       FORT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       USDP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       CSPR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        LDO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       FIDA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |         DC/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        RVN/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SWEAT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        PIX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      TRIBL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GMX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        EFI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        TON/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        XCV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        HFT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       ECOX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        AMB/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      AZERO/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        BDX/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        FLR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      ASTRA/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       SIMP/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        RPL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       HIGH/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        GFT/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       BLUR/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |       WAXL/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      FLOKI/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        SSV/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |     BLUR2S/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        FET/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      CFX2S/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |      SIDUS/USDT |         0 |           0.00 |           0.00 |             0.000 |           0.00 |             0:00 |     0     0     0     0 |
# |        NUM/USDT |         1 |          -0.25 |          -0.25 |            -0.336 |          -0.03 |          4:20:00 |     0     0     1     0 |
# |        GMT/USDT |         1 |          -0.28 |          -0.28 |            -0.395 |          -0.04 |          0:00:00 |     0     0     1     0 |
# |      BASIC/USDT |         1 |          -0.20 |          -0.20 |            -0.419 |          -0.04 |          0:00:00 |     0     0     1     0 |
# |      SQUAD/USDT |         1 |          -0.30 |          -0.30 |            -0.632 |          -0.06 |          0:00:00 |     0     0     1     0 |
# |       PEEL/USDT |         1 |          -0.20 |          -0.20 |            -0.978 |          -0.10 |          0:00:00 |     0     0     1     0 |
# |      CREDI/USDT |         1 |          -0.52 |          -0.52 |            -1.340 |          -0.13 |         18:25:00 |     0     0     1     0 |
# |       ORAI/USDT |         1 |          -0.31 |          -0.31 |            -1.535 |          -0.15 |          0:00:00 |     0     0     1     0 |
# |       INDI/USDT |         3 |           1.21 |           3.64 |            -1.700 |          -0.17 |          3:07:00 |     2     0     1  66.7 |
# |        ADS/USDT |         1 |          -0.31 |          -0.31 |            -1.971 |          -0.20 |          4:10:00 |     0     0     1     0 |
# |       KLAY/USDT |         1 |          -0.26 |          -0.26 |            -2.336 |          -0.23 |          7:25:00 |     0     0     1     0 |
# |       XETA/USDT |         2 |          -1.07 |          -2.14 |            -2.360 |          -0.24 |          0:00:00 |     0     0     2     0 |
# |       WEST/USDT |         2 |           0.01 |           0.02 |            -2.644 |          -0.26 |          5:20:00 |     1     0     1  50.0 |
# |       TRAC/USDT |         1 |          -0.75 |          -0.75 |            -3.471 |          -0.35 |          0:00:00 |     0     0     1     0 |
# |        KMA/USDT |         1 |          -0.87 |          -0.87 |            -4.136 |          -0.41 |          0:00:00 |     0     0     1     0 |
# |       CULT/USDT |         2 |          -0.84 |          -1.69 |            -5.186 |          -0.52 |          9:05:00 |     1     0     1  50.0 |
# |        EDG/USDT |         2 |           0.44 |           0.88 |            -5.885 |          -0.59 |          4:30:00 |     1     0     1  50.0 |
# |       VEGA/USDT |         3 |          -0.12 |          -0.37 |            -5.999 |          -0.60 |          9:47:00 |     2     0     1  66.7 |
# |       LOCG/USDT |         1 |          -1.65 |          -1.65 |            -6.006 |          -0.60 |          6:50:00 |     0     0     1     0 |
# |        CAS/USDT |         1 |          -2.43 |          -2.43 |            -6.032 |          -0.60 | 6 days, 15:50:00 |     0     0     1     0 |
# |       DATA/USDT |         1 |          -2.38 |          -2.38 |            -6.627 |          -0.66 |          6:35:00 |     0     0     1     0 |
# |       BAND/USDT |         1 |          -1.82 |          -1.82 |            -6.701 |          -0.67 |          8:35:00 |     0     0     1     0 |
# |      CUDOS/USDT |         3 |          -3.38 |         -10.13 |           -10.103 |          -1.01 |         20:47:00 |     2     0     1  66.7 |
# |   HOTCROSS/USDT |         2 |          -4.60 |          -9.20 |           -10.386 |          -1.04 |          7:20:00 |     1     0     1  50.0 |
# |       CARE/USDT |         4 |          -0.60 |          -2.41 |           -10.792 |          -1.08 |          0:51:00 |     1     0     3  25.0 |
# |        H2O/USDT |         1 |          -4.22 |          -4.22 |           -11.660 |          -1.17 |         22:50:00 |     0     0     1     0 |
# |        ENJ/USDT |         1 |          -4.31 |          -4.31 |           -11.798 |          -1.18 |         15:00:00 |     0     0     1     0 |
# |        CQT/USDT |         1 |          -1.67 |          -1.67 |           -11.957 |          -1.20 |          7:10:00 |     0     0     1     0 |
# |      ALEPH/USDT |         1 |          -3.81 |          -3.81 |           -12.489 |          -1.25 |   1 day, 6:25:00 |     0     0     1     0 |
# |       CELR/USDT |         1 |          -5.44 |          -5.44 |           -12.546 |          -1.25 | 3 days, 11:55:00 |     0     0     1     0 |
# |     WOMBAT/USDT |         1 |          -2.27 |          -2.27 |           -15.597 |          -1.56 |         17:40:00 |     0     0     1     0 |
# |       XCUR/USDT |         2 |          -2.68 |          -5.37 |           -17.145 |          -1.71 |          6:12:00 |     0     0     2     0 |
# |       GLCH/USDT |         2 |          -0.22 |          -0.44 |           -17.619 |          -1.76 |          1:55:00 |     1     0     1  50.0 |
# |        DPR/USDT |         1 |          -2.71 |          -2.71 |           -17.969 |          -1.80 |         13:35:00 |     0     0     1     0 |
# |      TRADE/USDT |         3 |          -1.73 |          -5.18 |           -19.700 |          -1.97 |          3:48:00 |     2     0     1  66.7 |
# |       MASK/USDT |         2 |          -3.17 |          -6.35 |           -20.861 |          -2.09 |  1 day, 12:42:00 |     1     0     1  50.0 |
# |        EQZ/USDT |         7 |          -0.88 |          -6.14 |           -21.331 |          -2.13 |          7:48:00 |     3     0     4  42.9 |
# |       BLOK/USDT |         2 |          -0.97 |          -1.93 |           -24.292 |          -2.43 |          8:10:00 |     1     0     1  50.0 |
# |       ROAR/USDT |         9 |           0.31 |           2.80 |           -24.563 |          -2.46 |          4:38:00 |     5     0     4  55.6 |
# |       ODDZ/USDT |         3 |          -2.26 |          -6.77 |           -25.855 |          -2.59 |          6:00:00 |     1     0     2  33.3 |
# |    HIBIRDS/USDT |        22 |          -0.00 |          -0.06 |           -27.108 |          -2.71 |          1:50:00 |    15     0     7  68.2 |
# |        ACH/USDT |         2 |          -1.16 |          -2.33 |           -28.417 |          -2.84 |          4:00:00 |     1     0     1  50.0 |
# |       VEED/USDT |         2 |          -3.89 |          -7.77 |           -29.299 |          -2.93 |         23:12:00 |     0     0     2     0 |
# |        SFP/USDT |         1 |          -7.15 |          -7.15 |           -31.264 |          -3.13 |         13:55:00 |     0     0     1     0 |
# |        CCD/USDT |         1 |          -3.93 |          -3.93 |           -31.717 |          -3.17 |  1 day, 22:35:00 |     0     0     1     0 |
# |      GRT2S/USDT |         2 |          -1.58 |          -3.15 |           -32.181 |          -3.22 |         10:58:00 |     1     0     1  50.0 |
# |        FTG/USDT |         3 |          -1.00 |          -2.99 |           -32.487 |          -3.25 |          8:23:00 |     1     0     2  33.3 |
# |       TAUM/USDT |        14 |           0.33 |           4.55 |           -35.671 |          -3.57 |          3:55:00 |     7     0     7  50.0 |
# |       POSI/USDT |         1 |          -4.91 |          -4.91 |           -35.729 |          -3.57 |         17:55:00 |     0     0     1     0 |
# |        FRR/USDT |         3 |          -0.99 |          -2.98 |           -37.651 |          -3.77 |          5:10:00 |     1     0     2  33.3 |
# |       USDJ/USDT |         4 |          -4.48 |         -17.92 |           -38.646 |          -3.86 |          0:32:00 |     2     0     2  50.0 |
# |       SLCL/USDT |         2 |          -1.79 |          -3.59 |           -38.772 |          -3.88 |         20:00:00 |     1     0     1  50.0 |
# |        JAR/USDT |        14 |          -0.24 |          -3.33 |           -41.984 |          -4.20 |          2:24:00 |     7     0     7  50.0 |
# |      COCOS/USDT |         1 |          -5.24 |          -5.24 |           -44.849 |          -4.48 |          5:25:00 |     0     0     1     0 |
# |     RANKER/USDT |         4 |          -1.60 |          -6.42 |           -46.011 |          -4.60 |         14:16:00 |     1     0     3  25.0 |
# |        LBP/USDT |         4 |          -0.78 |          -3.13 |           -46.787 |          -4.68 |          6:59:00 |     2     0     2  50.0 |
# |       SOLR/USDT |         3 |          -0.58 |          -1.73 |           -49.670 |          -4.97 |          7:28:00 |     2     0     1  66.7 |
# |       LACE/USDT |         4 |          -1.54 |          -6.18 |           -50.732 |          -5.07 |          9:52:00 |     2     0     2  50.0 |
# |       BRWL/USDT |         1 |          -5.96 |          -5.96 |           -50.897 |          -5.09 |          9:15:00 |     0     0     1     0 |
# |       OPCT/USDT |        13 |           0.03 |           0.36 |           -52.080 |          -5.21 |          4:02:00 |     9     0     4  69.2 |
# |       VSYS/USDT |         2 |          -2.31 |          -4.63 |           -54.874 |          -5.49 |          3:30:00 |     1     0     1  50.0 |
# |        MTS/USDT |         2 |          -4.62 |          -9.24 |           -64.634 |          -6.46 |         10:28:00 |     1     0     1  50.0 |
# |    HIMFERS/USDT |         9 |          -1.88 |         -16.92 |           -69.352 |          -6.94 |          2:10:00 |     4     0     5  44.4 |
# |        AOG/USDT |         3 |          -2.68 |          -8.04 |           -80.609 |          -8.06 |          6:55:00 |     2     0     1  66.7 |
# |       FEAR/USDT |         3 |          -2.99 |          -8.96 |           -88.006 |          -8.80 |         11:40:00 |     1     0     2  33.3 |
# |     CIX100/USDT |        12 |           0.85 |          10.24 |           -91.509 |          -9.15 |          2:23:00 |     9     0     3  75.0 |
# |        UNB/USDT |         4 |          -3.18 |         -12.73 |           -97.508 |          -9.75 |         12:09:00 |     0     0     4     0 |
# |       POLC/USDT |         2 |          -6.81 |         -13.63 |          -103.281 |         -10.33 |         11:42:00 |     1     0     1  50.0 |
# |       KING/USDT |         6 |          -1.45 |          -8.72 |          -104.560 |         -10.46 |          5:13:00 |     4     0     2  66.7 |
# |       HAKA/USDT |        12 |          -0.98 |         -11.76 |          -106.617 |         -10.66 |          4:21:00 |     8     0     4  66.7 |
# |        ACE/USDT |         4 |          -1.31 |          -5.25 |          -107.165 |         -10.72 |          2:19:00 |     3     0     1  75.0 |
# |       CLUB/USDT |         7 |          -0.72 |          -5.02 |          -114.101 |         -11.41 |          3:41:00 |     6     0     1  85.7 |
# |    WSIENNA/USDT |         2 |          -6.92 |         -13.84 |          -121.956 |         -12.20 |         14:58:00 |     0     0     2     0 |
# |       VELO/USDT |         8 |          -1.28 |         -10.26 |          -134.324 |         -13.43 |          5:48:00 |     6     0     2  75.0 |
# |        KAT/USDT |        15 |          -0.88 |         -13.22 |          -142.416 |         -14.24 |          1:34:00 |    10     0     5  66.7 |
# |        GLQ/USDT |        16 |          -0.40 |          -6.38 |          -152.865 |         -15.29 |          4:48:00 |     9     0     7  56.2 |
# |      ERTHA/USDT |         3 |          -6.66 |         -19.99 |          -167.798 |         -16.78 |          8:07:00 |     2     0     1  66.7 |
# |        IXS/USDT |         3 |          -6.34 |         -19.03 |          -171.812 |         -17.18 |         11:20:00 |     2     0     1  66.7 |
# |       ARNM/USDT |         8 |          -2.11 |         -16.92 |          -182.152 |         -18.22 |          2:05:00 |     3     0     5  37.5 |
# |      CFX2L/USDT |         1 |         -21.68 |         -21.68 |          -200.107 |         -20.01 |         13:10:00 |     0     0     1     0 |
# |           TOTAL |      1087 |           1.69 |        1832.01 |          4517.591 |         451.76 |          4:02:00 |   806     0   281  74.1 |
# ============================================================= ENTER TAG STATS ==============================================================
# |        TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# | Safe < EWO |      1043 |           1.66 |        1728.59 |          3991.384 |         399.14 |        4:05:00 |   766     0   277  73.4 |
# |  EWO < low |        44 |           2.35 |         103.42 |           526.207 |          52.62 |        2:41:00 |    40     0     4  90.9 |
# |      TOTAL |      1087 |           1.69 |        1832.01 |          4517.591 |         451.76 |        4:02:00 |   806     0   281  74.1 |
# ======================================================= EXIT REASON STATS ========================================================
# |        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
# |--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
# | trailing_stop_loss |     579 |    459     0   120  79.3 |           2.66 |        1539.1  |          4717.63  |         256.52 |
# |  Close > Offset Hi |     339 |    184     0   155  54.3 |          -0.77 |        -260.73 |         -2057.76  |         -43.46 |
# |                roi |     113 |    113     0     0   100 |           4.01 |         453.48 |          1736.85  |          75.58 |
# |  Close > Offset Lo |      50 |     49     0     1  98.0 |           2.75 |         137.47 |           432.042 |          22.91 |
# |         force_exit |       5 |      1     0     4  20.0 |          -3.11 |         -15.57 |          -130.543 |          -2.59 |
# |          stop_loss |       1 |      0     0     1     0 |         -21.74 |         -21.74 |          -180.629 |          -3.62 |
# ========================================================= LEFT OPEN TRADES REPORT ==========================================================
# |       Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
# |------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
# | GRT2S/USDT |         1 |           1.20 |           1.20 |             7.427 |           0.74 |        4:30:00 |     1     0     0   100 |
# |   CGG/USDT |         1 |          -2.03 |          -2.03 |           -14.629 |          -1.46 |        1:10:00 |     0     0     1     0 |
# |   RBP/USDT |         1 |          -3.08 |          -3.08 |           -22.002 |          -2.20 |        2:55:00 |     0     0     1     0 |
# |  GLCH/USDT |         1 |          -3.66 |          -3.66 |           -26.325 |          -2.63 |        1:15:00 |     0     0     1     0 |
# |  FEAR/USDT |         1 |          -8.01 |          -8.01 |           -75.014 |          -7.50 |       17:05:00 |     0     0     1     0 |
# |      TOTAL |         5 |          -3.11 |         -15.57 |          -130.543 |         -13.05 |        5:23:00 |     1     0     4  20.0 |
# =================== SUMMARY METRICS ====================
# | Metric                      | Value                  |
# |-----------------------------+------------------------|
# | Backtesting from            | 2023-01-01 00:00:00    |
# | Backtesting to              | 2023-02-25 00:00:00    |
# | Max open trades             | 6                      |
# |                             |                        |
# | Total/Daily Avg Trades      | 1087 / 19.76           |
# | Starting balance            | 1000 USDT              |
# | Final balance               | 5517.591 USDT          |
# | Absolute profit             | 4517.591 USDT          |
# | Total profit %              | 451.76%                |
# | CAGR %                      | 8365929.73%            |
# | Sortino                     | 39.38                  |
# | Sharpe                      | 56.03                  |
# | Calmar                      | 762.54                 |
# | Profit factor               | 1.66                   |
# | Expectancy                  | 0.17                   |
# | Trades per day              | 19.76                  |
# | Avg. daily profit %         | 8.21%                  |
# | Avg. stake amount           | 400.979 USDT           |
# | Total trade volume          | 435863.939 USDT        |
# |                             |                        |
# | Best Pair                   | HIVALHALLA/USDT 70.81% |
# | Worst Pair                  | CFX2L/USDT -21.68%     |
# | Best trade                  | UBX/USDT 21.48%        |
# | Worst trade                 | HIBIRDS/USDT -23.33%   |
# | Best day                    | 407.776 USDT           |
# | Worst day                   | -592.488 USDT          |
# | Days win/draw/lose          | 45 / 0 / 11            |
# | Avg. Duration Winners       | 2:29:00                |
# | Avg. Duration Loser         | 8:28:00                |
# | Rejected Entry signals      | 1130837                |
# | Entry/Exit Timeouts         | 0 / 0                  |
# |                             |                        |
# | Min balance                 | 990.521 USDT           |
# | Max balance                 | 5795.27 USDT           |
# | Max % of account underwater | 20.58%                 |
# | Absolute Drawdown (Account) | 20.58%                 |
# | Absolute Drawdown           | 1013.167 USDT          |
# | Drawdown high               | 3923.269 USDT          |
# | Drawdown low                | 2910.102 USDT          |
# | Drawdown Start              | 2023-02-08 14:05:00    |
# | Drawdown End                | 2023-02-10 10:30:00    |
# | Market change               | 70.95%                 |
# ========================================================
##### no bear detect #####
