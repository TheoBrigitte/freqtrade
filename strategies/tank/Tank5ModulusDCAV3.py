import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import os
import numpy as np
import pandas as pd
import warnings
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import List, Tuple, Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from typing import Optional
from functools import reduce
import warnings
import math
pd.options.mode.chained_assignment = None
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class Tank5ModulusDCAV3(IStrategy):

    '''
          ______   __          __              __    __   ______   __    __        __     __    __             ______            
     /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
    /$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
    $$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
    $$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
    $$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
    $$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
    $$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
     $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                       $$$$$$/                                               
                                                                                                                             
    '''          

    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.99
    locked_stoploss = {}
    timeframe = '5m'

    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(0, 5, default = 2 ,space='buy', optimize=True, load=True) # Of additional buys.
    max_dca_multiplier = DecimalParameter(low=1.1, high=4.0, default=1.2, decimals=1 ,space='buy', optimize=True, load=True)
    safety_order_reserve = IntParameter(4, 10, default=5.5, space='buy', optimize=True)
    filldelay = IntParameter(120, 360, default = 360 ,space='buy', optimize=True, load=True)
    max_entry_position_adjustment = max_epa.value
    ### Custom Functions
    # Modulus    
    peaks = IntParameter(30, 60, default=32, space='buy', optimize=True) ### initial smallest window
    bull_bear = IntParameter(80, 120, default=90, space='buy', optimize=True)
    trend = DecimalParameter(low=25, high=40, default=26.2, decimals=1 ,space='buy', optimize=True, load=True)
    volatility = DecimalParameter(low=30, high=50, default=34.6, decimals=1 ,space='buy', optimize=True, load=True)
    sensitivity = IntParameter(7, 15, default=12, space='buy', optimize=True, load=True)
    atr = IntParameter(3, 7, default=5, space='buy', optimize=True, load=True)
    window = IntParameter(12, 70, default=16, space='buy', optimize=True, load=True)
    mod = IntParameter(180, 200, default=196, space='buy', optimize=True, load=True)
    # AlphaTrend
    alphaDiv = IntParameter(30, 60, default=40, space='buy', optimize=True)
    alphaSig = IntParameter(30, 50, default=38, space='buy', optimize=True)
    alphaBuy = DecimalParameter(low=0.3, high=1.0, default=0.5, decimals=1 ,space='buy', optimize=True, load=True)
    alphaSell = DecimalParameter(low=-1.0, high=0.3, default=-0.5, decimals=1 ,space='sell', optimize=True, load=True)

    # Logic Selection
    use0 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use8 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use9 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)


    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True, load=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)

    locked_stoploss = {}

    minimal_roi = {
    }


    plot_config = {}


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


    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        # We need to leave most of the funds for possible further DCA orders
        if current_candle['sma'] < current_candle['200sma']:
            print(proposed_stake)
            calculated_stake = proposed_stake / (self.max_dca_multiplier.value + self.safety_order_reserve.value) 
            self.dp.send_msg(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
            logger.info(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
        else:
            # increase stake size in bullish enviroments
            calculated_stake = proposed_stake / (self.max_dca_multiplier.value)

        return calculated_stake 


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 
        # print(trade.pair, last_fill, current_time, count_of_entries)

        current_candle = dataframe.iloc[-1].squeeze()
        mod = current_candle['sma'] # shorter ma
        sma = current_candle['200sma'] # ref ma
        TP0 = current_candle['move_mean'] * 0.618
        TP0_5 = current_candle['move_mean']
        TP1 = current_candle['move_mean'] * 1.618
        TP2 = current_candle['move_mean'] * 2.618
        TP3 = current_candle['move_mean'] * 3.618
        display_profit = current_profit * 100
        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries}")
        # Take Profit if m00n
        if current_profit > TP1 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at average move fib%
            return -(trade.stake_amount / 4)
        if current_profit > TP2 and trade.nr_of_successful_exits == 1:
            # Take quarter of the profit at next fib%
            return -(trade.stake_amount / 3)
        if current_profit > TP3 and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%
            return -(trade.stake_amount / 2)
        if sma < mod:
            # Take Quick Profit if m00n
            if current_profit > TP3 and trade.nr_of_successful_exits == 0:
                return -trade.stake_amount
        if sma > mod:
            # Take Quick Profit if m00n
            if current_profit > TP2 and trade.nr_of_successful_exits == 0:
                return -trade.stake_amount            

            

        # Profit Based DCA   
        if trade.nr_of_successful_entries == self.max_epa.value:
            return None 
        if current_profit > -0.01:
            return None

        try:
            # This returns first order stake size 
            # Modify the following parameters to enable more levels or different buy size:
            # max_entry_position_adjustment = 3 
            # max_dca_multiplier = 3.5 

            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if (last_fill > self.filldelay.value) or (current_profit < -TP2):
                if (signal == 1 and current_profit < -0.01) or (current_profit < -TP1):
                    if count_of_entries == 1: 
                        stake_amount = stake_amount * 2
                    elif count_of_entries == 2:
                        stake_amount = stake_amount * 2
                    elif count_of_entries == 3:
                        stake_amount = stake_amount * 3
                    else:
                        stake_amount = stake_amount

                    return stake_amount
        except Exception as exception:
            return None

        return None
    

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        SLT1 = current_candle['move_mean']
        if trade_duration > 720 and trade_duration < 1080: 
            SL1 = current_candle['move_mean'] * 0.5
        if trade_duration > 1080 and trade_duration < 1440: 
            SL1 = current_candle['move_mean'] * 0.3
        else:    
            SL1 = current_candle['move_mean'] * 0.4

        SLT2 = current_candle['move_mean_x']
        SL2 = current_candle['move_mean_x'] - current_candle['move_mean']
        display_profit = current_profit * 100
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100
        # if len(self.locked_stoploss) > 0:
        #     print(self.locked_stoploss)

        if current_candle['max_l'] != 0:  # ignore stoploss if setting new highs
            if pair not in self.locked_stoploss:  # No locked stoploss for this pair yet
                if SLT2 is not None and current_profit > SLT2:
                    self.locked_stoploss[pair] = SL2
                    self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    return SL2
                elif SLT1 is not None and current_profit > SLT1:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    return SL1
                else:
                    return self.stoploss
            elif pair in self.locked_stoploss:  # Stoploss setting for each pair
                if SLT2 is not None and current_profit > SLT2:
                    self.locked_stoploss[pair] = SL2
                    self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    return SL2
                elif SLT1 is not None and current_profit > SLT1:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    return SL1
            else: # Stoploss has been locked for this pair
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% stoploss locked at {self.locked_stoploss[pair]:.4f}')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% stoploss locked at {self.locked_stoploss[pair]:.4f}')
                return self.locked_stoploss[pair]
        if current_profit < -.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                logger.info(f'*** {pair} *** Stoploss reset.')

        return self.stoploss



    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if exit_reason == 'roi' and (last_candle['max_l'] < 0.003):
            return False

        if exit_reason == 'trailing_stop_loss' and (last_candle['AlphaTrend'] > last_candle['AlphaTrendSig']):
            logger.info(f"{trade.pair} trailing stop temporarily released")
            # self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        # Handle freak events

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0")
            # self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            # self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bollinger_upperband'] = bollinger['upper']
        dataframe['bollinger_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)        
        
        pivots = pivot_points(dataframe)
        dataframe['pivot_lows'] = pivots['pivot_lows']
        dataframe['pivot_highs'] = pivots['pivot_highs']

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = (ta.MFI(dataframe, timeperiod=89) - 50) * 2
        dataframe["roc"] = ta.ROCR(dataframe, timeperiod=89)

        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["dpo"] = pta.dpo(dataframe['close'], length=40, centered=False)
        dataframe["dpo"] = dataframe["dpo"]
        # Williams R%
        dataframe['willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])

        # Modulus
        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['ha_wick'] = (heikinashi['high'] - heikinashi['close'])

        dataframe['HLC3'] = (heikinashi['high'] + heikinashi['low'] + heikinashi['close'])/3

        # WaveTrend using OHLC4 or HA close - 9/12
        ap = (0.333 * (heikinashi['high'] + heikinashi['low'] + heikinashi["close"]))

        dataframe['esa'] = ta.EMA(ap, timeperiod = 10)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 10)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 4)

        dataframe.loc[dataframe['wave_t1'] > 0, "wave_t1_UP"] = dataframe['wave_t1']
        dataframe.loc[dataframe['wave_t1'] < 0, "wave_t1_DN"] = dataframe['wave_t1']
        dataframe['wave_t1_UP'].ffill()
        dataframe['wave_t1_DN'].ffill()
        dataframe['wave_t1_MEAN_UP'] = dataframe['wave_t1_UP'].mean()
        dataframe['wave_t1_MEAN_DN'] = dataframe['wave_t1_DN'].mean()
        dataframe['wave_t1_UP_FIB'] = dataframe['wave_t1_MEAN_UP'] * 1.618
        dataframe['wave_t1_DN_FIB'] = dataframe['wave_t1_MEAN_DN'] * 1.618


        # 200 SMA and distance
        dataframe['200sma'] = ta.SMA(dataframe, timeperiod = 200)
        dataframe['200sma_dist'] = get_distance(heikinashi["close"], dataframe['200sma'])

        dataframe['sma'] = ta.EMA(dataframe, timeperiod=self.mod.value)
        dataframe['sma_pc'] = abs((dataframe['sma'] - dataframe['sma'].shift()) / dataframe['sma']) * 100
        dataframe['atr_pcnt'] = (qtpylib.atr(dataframe, window = self.atr.value)) / dataframe['ha_close']
        dataframe['modulation'] = 1 + (dataframe['sma_pc'] * self.trend.value) + (dataframe['atr_pcnt'] * self.volatility.value)

        # Set minimum and maximum window size
        min_window = self.peaks.value 
        max_window = self.bull_bear.value

        dataframe['order'] = (dataframe['modulation'] * self.peaks.value).round().fillna(self.bull_bear.value).astype(int)
        dataframe['order'] = np.where(dataframe['order'] > max_window, max_window, dataframe['order'])
        dataframe['order'] = np.where(dataframe['order'] < min_window, min_window, dataframe['order'])
        
        if not dataframe['order'].empty:
            order = dataframe['order'].iloc[-1]
        else:
            order = self.bear.value

        dataframe['zero'] = 0

        dataframe['max'] = dataframe["close"].rolling(order).max()/dataframe["close"] - 1
        dataframe['min'] = abs(dataframe["close"].rolling(order).min()/dataframe["close"] - 1)
        dataframe['mm_width'] = dataframe['max'] - dataframe['min']
        dataframe['atr_threshold'] = dataframe['atr_pcnt'].rolling(order).max()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max 
        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)

        dataframe['max_l'] = dataframe['OHLC4'].rolling(48).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(48).min() / dataframe['OHLC4'] - 1)

        dataframe['max_x'] = dataframe['OHLC4'].rolling(336).max() / dataframe['OHLC4'] - 1
        dataframe['min_x'] = abs(dataframe['OHLC4'].rolling(336).min() / dataframe['OHLC4'] - 1)


        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value) 
        rolling_max = rolling_window.max()
        rolling_min = rolling_window.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * 1.6
        dataframe['exit_mean'] = rolling_min * (1 + dataframe['move_mean'])
        dataframe['exit_mean_x'] = rolling_min * (1 + dataframe['move_mean_x'])
        dataframe['enter_mean'] = rolling_max * (1 - dataframe['move_mean'])
        dataframe['enter_mean_x'] = rolling_max * (1 - dataframe['move_mean_x'])
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=5) / dataframe['OHLC4'])
        dataframe['200sma_up'] = dataframe['200sma'] * (1 + dataframe['move_mean'])
        dataframe['200sma_dn'] = dataframe['200sma'] * (1 - dataframe['move_mean'])

        coeff = 1.618 #dataframe['move']
        AP = self.alphaSig.value

        dataframe['ATR'] = ta.SMA(ta.TRANGE(dataframe), timeperiod=AP)
        upT = dataframe['low'] - dataframe['ATR'] * coeff
        downT = dataframe['high'] + dataframe['ATR'] * coeff

        AlphaTrend = np.zeros(len(dataframe))
        AlphaTrend[0] = dataframe['close'][0]

        for i in range(1, len(dataframe)):
            if dataframe['rsi'].iloc[i-1] >= self.alphaDiv.value:
                AlphaTrend[i] = max(upT.iloc[i], AlphaTrend[i - 1])
            else:
                AlphaTrend[i] = min(downT.iloc[i], AlphaTrend[i - 1])

        dataframe['AlphaTrend'] = AlphaTrend
        dataframe['AlphaTrend'].ffill()
        dataframe['AlphaTrendSig'] = dataframe['AlphaTrend'].shift(3)
        dataframe['AlphaTrendDist'] = PC(dataframe, dataframe['AlphaTrend'], dataframe['AlphaTrendSig'])


        initialize_divergences_lists(dataframe)
        add_divergences(dataframe, 'rsi')
        add_divergences(dataframe, 'stoch')
        add_divergences(dataframe, 'roc')
        add_divergences(dataframe, 'uo')
        add_divergences(dataframe, 'ao')
        add_divergences(dataframe, 'cci')
        add_divergences(dataframe, 'obv')
        add_divergences(dataframe, 'mfi')
        add_divergences(dataframe, 'adx')
        add_divergences(dataframe, 'wave_t1')
        add_divergences(dataframe, 'AlphaTrend')
        add_divergences(dataframe, 'dpo')
        add_divergences(dataframe, 'move')

        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        full_send1 = (
                (self.use0.value == True) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["willr14"] < -50) &
                (df["move"] < df['move_mean']) &
                (df["AlphaTrendDist"] == 0) &
                (df["OHLC4"] < df['200sma_dn']) &
                (df["OHLC4"].shift(48) < df['200sma_up'].shift(48)) &
                (df["wave_t1"] < 0) &
                (df['volume'] > 0)   # Make sure Volume is not 0
        )
            
        df.loc[full_send1, 'enter_long'] = 1
        df.loc[full_send1, 'enter_tag'] = 'Full Send 1'

        full_send2 = (
                (self.use1.value == True) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["willr14"] < -50) &
                (df["AlphaTrendDist"] > self.alphaBuy.value) &
                # (df["OHLC4"] < df['200sma']) &
                (df['volume'] > 0)   # Make sure Volume is not 0
        )
        df.loc[full_send2, 'enter_long'] = 1
        df.loc[full_send2, 'enter_tag'] = 'Full Send 2'

        full_send3 = (
                (self.use2.value == True) &
                (df["willr14"] < -85) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df['order'] == self.bull_bear.value) &
                (df["wave_t1"] < df["wave_t1_DN_FIB"]) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send3, 'enter_long'] = 1
        df.loc[full_send3, 'enter_tag'] = 'Full Send 3'

        full_send4 = (
                (self.use3.value == True) &
                (df["willr14"] < -85) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["wave_t1"] < df["wave_t1_DN_FIB"]) &
                (df['order'] < self.bull_bear.value) &
                (df['order'] > (self.bull_bear.value / 2)) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send4, 'enter_long'] = 1
        df.loc[full_send4, 'enter_tag'] = 'Full Send 4'

        full_send5 = (
                (self.use4.value == True) &
                (df["willr14"] < -85) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["wave_t1"] < df["wave_t1_MEAN_DN"]) &
                (df['order'] <= (self.bull_bear.value / 2)) &
                (df['order'] >= self.peaks.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send5, 'enter_long'] = 1
        df.loc[full_send5, 'enter_tag'] = 'Full Send 5'

        full_send6 = (
                (self.use5.value == True) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["willr14"] < -85) &
                (df["sma"] < df['200sma']) &
                (df["sma"] < df['sma'].shift()) &
                (df["wave_t1"] < df["wave_t1_MEAN_DN"]) &
                (df['order'] <= (self.bull_bear.value / 2)) &
                (df['order'] >= self.peaks.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send6, 'enter_long'] = 1
        df.loc[full_send6, 'enter_tag'] = 'Full Send 6'

        full_send7 = (              
                (self.use6.value == True) &                
                (df["sma"] < df['200sma']) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["willr14"] < -85) &
                (df["sma"].shift() > df['200sma'].shift()) &
                (df["OHLC4"] < df['200sma_dn']) &
                (df["wave_t1"] < df["wave_t1_MEAN_DN"]) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send7, 'enter_long'] = 1
        df.loc[full_send7, 'enter_tag'] = 'Full Send 7'



        full_send8 = (           
                (self.use7.value == True) &                   
                (df["sma"] > df['200sma']) &
                (df['total_bullish_divergences'].shift() > 0) &
                (df["willr14"] < -85) &
                (df["sma"].shift() < df['200sma'].shift()) &
                (df["OHLC4"] < df['200sma']) &
                (df["wave_t1"] < df["wave_t1_MEAN_DN"]) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send8, 'enter_long'] = 1
        df.loc[full_send8, 'enter_tag'] = 'Full Send 8'

        # is_entry = ( 
        #     (df['entry'].shift() == 1) &
        #     (df['entry'] !=1 )
        #     )
        # df.loc[is_entry, 'enter_long'] = 1


        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        profit_taker1 = (
                (self.use8.value == True) &
                (df["willr14"] > -50) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df["AlphaTrendDist"] < self.alphaSell.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker1, 'exit_long'] = 1
        df.loc[profit_taker1, 'exit_tag'] = 'Profit Taker 1'

        profit_taker2 = (
                (self.use9.value == True) &
                (df["willr14"] > -15) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df['order'] == self.bull_bear.value) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker2, 'exit_long'] = 1
        df.loc[profit_taker2, 'exit_tag'] = 'Profit Taker 2'

        profit_taker3 = (
                (self.use10.value == True) &
                (df["willr14"] > -15) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &
                (df["AlphaTrendDist"] < self.alphaSell.value) &
                (df['order'] < self.bull_bear.value) &
                (df['order'] > (self.bull_bear.value / 2)) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker3, 'exit_long'] = 1
        df.loc[profit_taker3, 'exit_tag'] = 'Profit Taker 3'

        profit_taker4 = (
                (self.use11.value == True) &
                (df["willr14"] > -15) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &
                (df['order'] <= (self.bull_bear.value / 2)) &
                (df['order'] >= self.peaks.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker4, 'exit_long'] = 1
        df.loc[profit_taker4, 'exit_tag'] = 'Profit Taker 4'

        profit_taker5 = (
                (self.use12.value == True) &
                (df["willr14"] > -15) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &
                (df['order'] <= (self.bull_bear.value / 2)) &
                (df['order'] >= self.peaks.value) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker5, 'exit_long'] = 1
        df.loc[profit_taker5, 'exit_tag'] = 'Profit Taker 5'

        profit_taker6 = (
                (self.use13.value == True) &
                (df['total_bearish_divergences'].shift() > 0) &
                (df["willr14"] > -15) &
                (df["200sma"] > df["exit_mean_x"]) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &
                (df["close"] > df["exit_mean_x"]) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker6, 'exit_long'] = 1
        df.loc[profit_taker6, 'exit_tag'] = 'Profit Taker 6'

        # is_exiting = ( 
        #     (df['exiting'].shift() == 1) &
        #     (df['exiting'] !=1 )
        #     )
        # df.loc[is_exiting, 'exit_long'] = 1


        return df


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def get_distance(p1, p2):
    return (p1) - (p2)

def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc

def two_bands_check(self, dataframe):
    check = (
        (dataframe['low'] < dataframe['kc_lowerband']) & (dataframe['high'] > dataframe['kc_upperband'])
    )
    return ~check

def initialize_divergences_lists(dataframe: pd.DataFrame):
    
    dataframe["total_bullish_divergences"] = np.nan
    dataframe["total_bullish_divergences_count"] = 0
    dataframe["total_bullish_divergences_names"] = ''
    
    dataframe["total_bearish_divergences"] = np.nan
    dataframe["total_bearish_divergences_count"] = 0
    dataframe["total_bearish_divergences_names"] = ''

def add_divergences(dataframe: DataFrame, indicator: str):
    (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = divergence_finder_dataframe(dataframe, indicator)
    dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
    dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences

def divergence_finder_dataframe(dataframe: DataFrame, indicator_source: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
    bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
    bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
    bullish_divergences = np.empty(len(dataframe['close'])) * np.nan
    low_iterator = []
    high_iterator = []

    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        if np.isnan(row.pivot_lows):
            low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
        else:
            low_iterator.append(index)
        if np.isnan(row.pivot_highs):
            high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
        else:
            high_iterator.append(index)

    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

        bearish_occurence = bearish_divergence_finder(dataframe,
            dataframe[indicator_source],
            high_iterator,
            index)

        if bearish_occurence is not None:
            (prev_pivot , current_pivot) = bearish_occurence 
            bearish_prev_pivot = dataframe.loc[prev_pivot, 'close']
            bearish_current_pivot = dataframe.loc[current_pivot, 'close']
            bearish_ind_prev_pivot = dataframe.loc[prev_pivot, indicator_source]
            bearish_ind_current_pivot = dataframe.loc[current_pivot, indicator_source]
            length = current_pivot - prev_pivot
            bearish_lines_index = 0
            can_exist = True
            while(True):
                can_draw = True
                if bearish_lines_index <= len(bearish_lines):
                    bearish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                actual_bearish_lines = bearish_lines[bearish_lines_index]
                for i in range(length + 1):
                    point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                    indicator_point =  bearish_ind_prev_pivot + (bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                    if i != 0 and i != length:
                        if (point <= dataframe.loc[prev_pivot + i, 'close'] 
                        or indicator_point <= dataframe.loc[prev_pivot + i, indicator_source]):
                            can_exist = False
                    if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                        can_draw = False
                if not can_exist:
                    break
                if can_draw:
                    for i in range(length + 1):
                        actual_bearish_lines[prev_pivot + i] = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                    break
                bearish_lines_index = bearish_lines_index + 1
            if can_exist:
                bearish_divergences[index] = row.close
                dataframe.loc[index, "total_bearish_divergences"] = row.close
                if index > 30:
                    dataframe.loc[index-30, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index-30, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

        bullish_occurence = bullish_divergence_finder(dataframe,
            dataframe[indicator_source],
            low_iterator,
            index)
        
        if bullish_occurence is not None:
            (prev_pivot , current_pivot) = bullish_occurence
            bullish_prev_pivot = dataframe.loc[prev_pivot, 'close']
            bullish_current_pivot = dataframe.loc[current_pivot, 'close']
            bullish_ind_prev_pivot = dataframe.loc[prev_pivot, indicator_source]
            bullish_ind_current_pivot = dataframe.loc[current_pivot, indicator_source]
            length = current_pivot - prev_pivot
            bullish_lines_index = 0
            can_exist = True
            while(True):
                can_draw = True
                if bullish_lines_index <= len(bullish_lines):
                    bullish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                actual_bullish_lines = bullish_lines[bullish_lines_index]
                for i in range(length + 1):
                    point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                    indicator_point =  bullish_ind_prev_pivot + (bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                    if i != 0 and i != length:
                        if (point >= dataframe.loc[prev_pivot + i, 'close'] 
                        or indicator_point >= dataframe.loc[prev_pivot + i, indicator_source]):
                            can_exist = False
                    if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                        can_draw = False
                if not can_exist:
                    break
                if can_draw:
                    for i in range(length + 1):
                        actual_bullish_lines[prev_pivot + i] = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                    break
                bullish_lines_index = bullish_lines_index + 1
            if can_exist:
                bullish_divergences[index] = row.close
                dataframe.loc[index, "total_bullish_divergences"] = row.close
                if index > 30:
                    dataframe.loc[index-30, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index-30, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'
    
    return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)


def bearish_divergence_finder(dataframe, indicator, high_iterator, index):
    if high_iterator[index] == index:
        current_pivot = high_iterator[index]
        occurences = list(dict.fromkeys(high_iterator))
        current_index = occurences.index(high_iterator[index])
        for i in range(current_index-1,current_index-6,-1):            
            prev_pivot = occurences[i]
            if np.isnan(prev_pivot):
                return
            if ((dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
            or (dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                return (prev_pivot , current_pivot)
    return None

def bullish_divergence_finder(dataframe, indicator, low_iterator, index):
    if low_iterator[index] == index:
        current_pivot = low_iterator[index]
        occurences = list(dict.fromkeys(low_iterator))
        current_index = occurences.index(low_iterator[index])
        for i in range(current_index-1,current_index-6,-1):
            prev_pivot = occurences[i]
            if np.isnan(prev_pivot):
                return 
            if ((dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
            or (dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                return (prev_pivot, current_pivot)
    return None

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source: int =1) -> DataFrame:
    high_source = None
    low_source = None

    if pivot_source == 1:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == 0:
        high_source = 'high'
        low_source = 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()
    
    # find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()

    # find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    is_greater = True
    is_less = True
    if (getattr(current_value, high_source) < getattr(left, high_source) or
        getattr(current_value, high_source) < getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) > getattr(left, low_source) or
        getattr(current_value, low_source) > getattr(right, low_source)):
        is_less = False
    return (is_greater, is_less)

def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

