# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from typing import Optional
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

# @Rallipanos # changes by IcHiAT


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class Auto_EI_t4c0s(IStrategy):

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

    ### 3-13-2023 ###
    # WIP - V1
    # No hyper-opt using just defaults from orig EI3 strat, you need to try and hyper-opt this file -
    # Will suggest adding in an roi table.
    ###


    # Buy hyperspace params:
    buy_params = {
        "atr_length": 23,
        "b01": 0.5,
        "b02": 0.5,
        "b03": 0.5,
        "b04": 0.5,
        "b05": 0.5,
        "b06": 0.5,
        "base_nb_candles_buy": 17,
        "buy_adx": 28,
        "buy_ema_cofi": 0.969,
        "buy_ewo_high": 5.687,
        "buy_fastd": 23,
        "buy_fastk": 30,
        "fib_dn": 2.5,
        "fib_up": 9.5,
        "increment": 1.0007,
        "lambo2_ema_14_factor": 0.967,
        "lambo2_rsi_14_limit": 27,
        "lambo2_rsi_4_limit": 32,
        "low_offset": 0.995,
        "mean_dn": 1.6,
        "mean_up": 3.3,
        "rsi_buy": 54,
        "window": 30,
        "x01": 4.2,
        "x02": 1.3,
        "x03": 1.2,
        "x04": 1.3,
        "x05": 4.2,
        "z01": 2.1,
        "zero_dn": 4.1,
        "zero_up": 6.6,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 14,
        "day1": 1,
        "day2": 4,
        "day3": 5,
        "day4": 8,
        "fib_dns": 2.3,
        "fib_ups": 1.6,
        "high_offset": 1.012,
        "high_offset_2": 1.014,
        "mean_dns": 2.2,
        "mean_ups": 5.4,
        "moon": 0.004,
        "s01": 0.5,
        "s02": 0.5,
        "s03": 0.5,
        "s04": 0.5,
        "s05": 0.5,
        "s06": 0.5,
        "unclog1": 0.15,
        "unclog2": 0.16,
        "unclog3": 0.12,
        "unclog4": 0.18,
        "y01": 3.6,
        "y02": 3.4,
        "y03": 2.6,
        "zero_dns": 7.3,
        "zero_ups": 1.8,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.243,
        "46": 0.035,
        "127": 0.015,
        "237": 0
    }


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


    # Stoploss - Unclog
    stoploss = -0.99
    moon = DecimalParameter(0.001, 0.004, default=0.35,  decimals=3, space='sell', optimize=True)
    unclog1 = DecimalParameter(0.1, 0.2, default=0.04,  decimals=2, space='sell', optimize=True)
    unclog2 = DecimalParameter(0.08, 0.2, default=0.04,  decimals=2, space='sell', optimize=True)
    unclog3 = DecimalParameter(0.10, 0.2, default=0.04,  decimals=2, space='sell', optimize=True)
    unclog4 = DecimalParameter(0.14, 0.2, default=0.04,  decimals=2, space='sell', optimize=True)
    day1 = IntParameter(1, 4, default=1, space='sell', optimize=True)
    day2 = IntParameter(2, 5, default=2, space='sell', optimize=True)
    day3 = IntParameter(3, 6, default=3, space='sell', optimize=True)
    day4 = IntParameter(4, 10, default=4, space='sell', optimize=True)


    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    locked_stoploss = {}

    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    window = IntParameter(12, 70, default=48, space='buy', optimize=True)

    #cofi
    is_optimize_cofi = True
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    atr_length = IntParameter(10, 30, default=14, space='buy', optimize=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)

    ###  Buy Weight Mulitpliers ###
    x01 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x02 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x03 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x04 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    x05 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)

    ###  Sell Weight Mulitpliers ###
    y01 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y02 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)
    y03 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)

    ###  General Weight Mulitpliers ###
    z01 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)
    z02 = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='sell', optimize=True)

    ### Entry / Exit Thresholds ###
    b01 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)
    b02 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)
    b03 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)
    b04 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)
    b05 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)
    b06 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='buy', optimize=True)

    s01 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)
    s02 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)
    s03 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)
    s04 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)
    s05 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)
    s06 = DecimalParameter(0.0, 1.10, default=0.5, decimals=2, space='sell', optimize=True)

    fib_dn = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)
    mean_dn = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)
    zero_dn = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)
    zero_up = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)
    mean_up = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)
    fib_up = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='buy', optimize=True)

    fib_dns = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)
    mean_dns = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)
    zero_dns = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)
    zero_ups = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)
    mean_ups = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)
    fib_ups = DecimalParameter(1.0, 10.0, default=2.5, decimals=1, space='sell', optimize=True)


    use_custom_stoploss = True
    process_only_new_candles = True
    # Custom Entry
    last_entry_price = None

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        SLT1 = current_candle['move_mean']
        SL1 = current_candle['move_mean'] * 0.5
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
            else:  # Stoploss has been locked for this pair
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

        # Handle freak events

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0")
            self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than X days.
        if current_profit < -self.unclog4.value and (current_time - trade.open_date_utc).days >= self.day4.value:
            return 'unclog 4'
        if current_profit < -self.unclog3.value and (current_time - trade.open_date_utc).days >= self.day3.value:
            return 'unclog 3'
        if current_profit < -self.unclog2.value and (current_time - trade.open_date_utc).days >= self.day2.value:
            return 'unclog 2'
        if current_profit < -self.unclog1.value and (current_time - trade.open_date_utc).days >= self.day1.value:
            return 'unclog 1'
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    ## Optional order time in force.
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}
    # Optimal timeframe for the strategy
    timeframe = '5m'

    position_adjustment_enable = False
    process_only_new_candles = True
    startup_candle_count = 200


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value)
        dataframe['ma_hi'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset.value)
        dataframe['ma_hi_2'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.high_offset_2.value)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        # HMA-BUY SQUEEZE
        dataframe['HMA_SQZ'] = (((dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] - dataframe['hma_50']) 
            / dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']) * 100)


        dataframe['zero'] = 0
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe.loc[dataframe['EWO'] > 0, "EWO_UP"] = dataframe['EWO']
        dataframe.loc[dataframe['EWO'] < 0, "EWO_DN"] = dataframe['EWO']
        dataframe['EWO_UP'].ffill()
        dataframe['EWO_DN'].ffill()
        dataframe['EWO_MEAN_UP'] = dataframe['EWO_UP'].mean()
        dataframe['EWO_MEAN_DN'] = dataframe['EWO_DN'].mean()
        dataframe['EWO_UP_FIB'] = dataframe['EWO_MEAN_UP'] * 1.618
        dataframe['EWO_DN_FIB'] = dataframe['EWO_MEAN_DN'] * 1.618

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        #lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

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

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window_x = dataframe['OHLC4'].rolling(200)
        rolling_max_x = rolling_window_x.max()
        rolling_min_x = rolling_window_x.min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value_x = rolling_window_x.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move_l'] = ptp_value_x / dataframe['OHLC4']
        dataframe['move_mean_l'] = dataframe['move_l'].mean()
        dataframe['move_mean_xl'] = dataframe['move_l'].mean() * 1.6
        dataframe['exit_mean_l'] = rolling_min_x * (1 + dataframe['move_mean_l'])
        dataframe['exit_mean_xl'] = rolling_min_x * (1 + dataframe['move_mean_xl'])
        dataframe['enter_mean_l'] = rolling_max_x * (1 - dataframe['move_mean_l'])
        dataframe['enter_mean_xL'] = rolling_max_x * (1 - dataframe['move_mean_xl'])

        ### Buying Weights & Signals ###
        dataframe.loc[(dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)), 'buy0'] = 1
        dataframe.loc[(dataframe['close'] > (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)), 'buy0'] = 0
        dataframe.loc[(dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)), 'buy1'] = 1
        dataframe.loc[(dataframe['rsi_4'] > int(self.lambo2_rsi_4_limit.value)), 'buy1'] = 0
        dataframe.loc[(dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)), 'buy2'] = 1
        dataframe.loc[(dataframe['rsi_14'] > int(self.lambo2_rsi_14_limit.value)), 'buy2'] = 0
        dataframe.loc[(dataframe['atr_pcnt'] > dataframe['min_l']), 'buy3'] = 1
        dataframe.loc[(dataframe['atr_pcnt'] < dataframe['min_l']), 'buy3'] = 0 
        dataframe.loc[(dataframe['rsi']<self.rsi_buy.value), 'buy4'] = 1
        dataframe.loc[(dataframe['rsi']>self.rsi_buy.value), 'buy4'] = 0

        dataframe['lambo_weight'] = (
            (dataframe['buy0']+dataframe['buy1']+dataframe['buy2']+dataframe['buy3']+dataframe['buy4'])/5) * self.x01.value

        dataframe.loc[(dataframe['rsi_fast'] < 35), 'buy10'] = 1
        dataframe.loc[(dataframe['rsi_fast'] > 35), 'buy10'] = 0
        dataframe.loc[(dataframe['close'] < dataframe['ma_lo']), 'buy11'] = 1
        dataframe.loc[(dataframe['close'] > dataframe['ma_lo']), 'buy11'] = 0
        dataframe.loc[(dataframe['close'] < dataframe['enter_mean_x']), 'buy12'] = 1
        dataframe.loc[(dataframe['close'] > dataframe['enter_mean_x']), 'buy12'] = 0
        dataframe.loc[(dataframe['close'].shift() < dataframe['enter_mean_x'].shift()), 'buy13'] = 1
        dataframe.loc[(dataframe['close'].shift() > dataframe['enter_mean_x'].shift()), 'buy13'] = 0 
        dataframe.loc[(dataframe['rsi'] < self.rsi_buy.value), 'buy14'] = 1
        dataframe.loc[(dataframe['rsi'] > self.rsi_buy.value), 'buy14'] = 0
        dataframe.loc[(dataframe['atr_pcnt'] > dataframe['min']), 'buy15'] = 1
        dataframe.loc[(dataframe['atr_pcnt'] < dataframe['min']), 'buy15'] = 0
        dataframe.loc[(dataframe['EWO'] > dataframe['EWO_MEAN_UP']), 'buy16'] = 1
        dataframe.loc[(dataframe['EWO'] < dataframe['EWO_MEAN_UP']), 'buy16'] = 0 

        dataframe['buy1ewo_weight'] = (
            (dataframe['buy10']+dataframe['buy11']+dataframe['buy12']+dataframe['buy13']
                +dataframe['buy14']+dataframe['buy15']+dataframe['buy16'])/7) * self.x02.value

        dataframe.loc[(dataframe['rsi_fast'] < 35), 'buy20'] = 1
        dataframe.loc[(dataframe['rsi_fast'] > 35), 'buy20'] = 0
        dataframe.loc[(dataframe['close'] < dataframe['ma_lo']), 'buy21'] = 1
        dataframe.loc[(dataframe['close'] > dataframe['ma_lo']), 'buy21'] = 0
        dataframe.loc[(dataframe['EWO'] < dataframe['EWO_DN_FIB']), 'buy22'] = 1
        dataframe.loc[(dataframe['EWO'] > dataframe['EWO_DN_FIB']), 'buy22'] = 0
        dataframe.loc[(dataframe['atr_pcnt'] > dataframe['min']), 'buy23'] = 1
        dataframe.loc[(dataframe['atr_pcnt'] < dataframe['min']), 'buy23'] = 0

        dataframe['buy2ewo_weight'] = (
            (dataframe['buy20']+dataframe['buy21']+dataframe['buy22']+dataframe['buy23'])/4) * self.x03.value


        dataframe.loc[(dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value), 'buy30'] = 1
        dataframe.loc[(dataframe['open'] > dataframe['ema_8'] * self.buy_ema_cofi.value), 'buy30'] = 0
        dataframe['buy31'] = 0 
        dataframe.loc[qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']), 'buy31'] = 1
        dataframe.loc[(dataframe['fastk'] < self.buy_fastk.value), 'buy32'] = 1
        dataframe.loc[(dataframe['fastk'] > self.buy_fastk.value), 'buy32'] = 0
        dataframe.loc[(dataframe['fastd'] < self.buy_fastd.value), 'buy33'] = 1
        dataframe.loc[(dataframe['fastd'] > self.buy_fastd.value), 'buy33'] = 0
        dataframe.loc[(dataframe['adx'] > self.buy_adx.value), 'buy34'] = 1
        dataframe.loc[(dataframe['adx'] < self.buy_adx.value), 'buy34'] = 0
        dataframe.loc[(dataframe['EWO'] > dataframe['EWO_MEAN_UP']), 'buy35'] = 1
        dataframe.loc[(dataframe['EWO'] < dataframe['EWO_MEAN_UP']), 'buy35'] = 0
        dataframe.loc[(dataframe['atr_pcnt'] > dataframe['min']), 'buy36'] = 1
        dataframe.loc[(dataframe['atr_pcnt'] < dataframe['min']), 'buy36'] = 0


        dataframe['cofi_weight'] = (
            (dataframe['buy30']+dataframe['buy31']+dataframe['buy32']+dataframe['buy33']+dataframe['buy34']+dataframe['buy35']+dataframe['buy36'])/7) * self.x04.value

        dataframe['buy_weight'] = ta.SMA(((dataframe['lambo_weight'] + dataframe['buy1ewo_weight'] + dataframe['buy2ewo_weight'] + dataframe['cofi_weight']) / 4) * self.x05.value, timeperiod=5)

        ### General Indicators ###
        dataframe.loc[(dataframe['fastk'] < self.buy_fastk.value), 'gen1'] = 1
        dataframe.loc[(dataframe['fastk'] > self.buy_fastk.value), 'gen1'] = -1
        dataframe.loc[(dataframe['fastd'] < self.buy_fastd.value), 'gen2'] = 1
        dataframe.loc[(dataframe['fastd'] > self.buy_fastd.value), 'gen2'] = -1
        dataframe.loc[(dataframe['adx'] > self.buy_adx.value), 'gen3'] = 1
        dataframe.loc[(dataframe['adx'] < self.buy_adx.value), 'gen3'] = -1
        dataframe['gen4'] = 0
        dataframe.loc[qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']), 'gen4'] = 1
        dataframe.loc[qtpylib.crossed_below(dataframe['fastk'], dataframe['fastd']), 'gen4'] = -1

        # Distance from min and max summed for different ranges. Selling it will go negative.
        dataframe['gen5'] = (dataframe['max']+dataframe['max_l']+dataframe['max_x']) - (dataframe['min']+dataframe['min_l']+dataframe['min_x'])

        dataframe['gen6'] = 0
        dataframe.loc[qtpylib.crossed_above(dataframe['ema_8'], dataframe['ema_14']), 'gen6'] = 1
        dataframe.loc[qtpylib.crossed_below(dataframe['ema_8'], dataframe['ema_14']), 'gen6'] = -1

        dataframe['general_weight'] = ((dataframe['gen1']+dataframe['gen2']+dataframe['gen3']+dataframe['gen4']+dataframe['gen5']+dataframe['gen6'])/6) 

        dataframe.loc[(dataframe['general_weight'] > 0), 'gen_buy'] = dataframe['general_weight'] * (1 + dataframe['move']) 
        dataframe.loc[(dataframe['general_weight'] < 0), 'gen_buy'] = 0
        dataframe.loc[(dataframe['general_weight'] < 0), 'gen_sell'] = abs(dataframe['general_weight'] * (1 + dataframe['move'])) 
        dataframe.loc[(dataframe['general_weight'] > 0), 'gen_sell'] = 0     


        ### SELLING Weights & Signals ###
        dataframe.loc[(dataframe['close'] > dataframe['hma_50']), 'sell0'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['hma_50']), 'sell0'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['ma_hi_2']), 'sell1'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['ma_hi_2']), 'sell1'] = 0
        dataframe.loc[(dataframe['max_l'] != 0), 'sell2'] = 1
        dataframe.loc[(dataframe['max_l'] == 0), 'sell2'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['exit_mean_x']), 'sell3'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['exit_mean_x']), 'sell3'] = 0
        dataframe.loc[(dataframe['rsi'] > 50), 'sell4'] = 1
        dataframe.loc[(dataframe['rsi'] < 50), 'sell4'] = 0
        dataframe.loc[(dataframe['rsi_fast'] > dataframe['rsi_slow']), 'sell5'] = 1
        dataframe.loc[(dataframe['rsi_fast'] < dataframe['rsi_slow']), 'sell5'] = 0

        dataframe['hi2_weight'] = (
            (dataframe['sell0']+dataframe['sell1']+dataframe['sell2']+dataframe['sell3']+dataframe['sell4']+dataframe['sell5'])/6) * self.y01.value

        dataframe.loc[(dataframe['close'] < dataframe['hma_50']), 'sell10'] = 1
        dataframe.loc[(dataframe['close'] > dataframe['hma_50']), 'sell10'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['ma_hi']), 'sell11'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['ma_hi']), 'sell11'] = 0
        dataframe.loc[(dataframe['max_l'] != 0), 'sell12'] = 1
        dataframe.loc[(dataframe['max_l'] == 0), 'sell12'] = 0
        dataframe.loc[(dataframe['rsi_fast'] > dataframe['rsi_slow']), 'sell13'] = 1
        dataframe.loc[(dataframe['rsi_fast'] < dataframe['rsi_slow']), 'sell13'] = 0

        dataframe['hi_weight'] = (
            (dataframe['sell10']+dataframe['sell11']+dataframe['sell12']+dataframe['sell13'])/4) * self.y02.value

        dataframe['sell_weight'] = ta.SMA(((dataframe['hi_weight'] + dataframe['hi2_weight']) / 2) * self.y03.value, timeperiod=5)


        dataframe['buy_decision'] = dataframe['buy_weight'] - dataframe['sell_weight']
        dataframe['sell_decision'] = dataframe['sell_weight'] - dataframe['buy_weight']

        dataframe['Gen Buy Above'] = 0
        dataframe.loc[(dataframe['gen_buy'] > self.b01.value), 'Gen Buy Above'] = 1
        dataframe.loc[(dataframe['gen_buy'] > self.b02.value), 'Gen Buy Above'] = 1
        dataframe.loc[(dataframe['gen_buy'] > self.b03.value), 'Gen Buy Above'] = 1
        dataframe.loc[(dataframe['gen_buy'] > self.b04.value), 'Gen Buy Above'] = 1
        dataframe.loc[(dataframe['gen_buy'] > self.b05.value), 'Gen Buy Above'] = 1
        dataframe.loc[(dataframe['gen_buy'] > self.b06.value), 'Gen Buy Above'] = 1

        dataframe['Gen Sell Above'] = 0
        dataframe.loc[(dataframe['gen_sell'] > self.s01.value), 'Gen Sell Above'] = 1
        dataframe.loc[(dataframe['gen_sell'] > self.s02.value), 'Gen Sell Above'] = 1
        dataframe.loc[(dataframe['gen_sell'] > self.s03.value), 'Gen Sell Above'] = 1
        dataframe.loc[(dataframe['gen_sell'] > self.s04.value), 'Gen Sell Above'] = 1
        dataframe.loc[(dataframe['gen_sell'] > self.s05.value), 'Gen Sell Above'] = 1
        dataframe.loc[(dataframe['gen_sell'] > self.s06.value), 'Gen Sell Above'] = 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ewo_fib_dn = (
                (dataframe['EWO'] < dataframe['EWO_DN_FIB']) &
                (dataframe['buy_decision'] > self.fib_dn.value) &
                (dataframe['gen_buy'] > self.b01.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_fib_dn, 'enter_long'] = 1
        dataframe.loc[ewo_fib_dn, 'enter_tag'] = 'ewo_fib_dn'

        ewo_mean_dn = (
                (dataframe['EWO'] > dataframe['EWO_DN_FIB']) &
                (dataframe['EWO'] < dataframe['EWO_MEAN_DN']) &
                (dataframe['buy_decision'] > self.mean_dn.value) &
                (dataframe['gen_buy'] > self.b02.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_mean_dn, 'enter_long'] = 1
        dataframe.loc[ewo_mean_dn, 'enter_tag'] = 'ewo_mean_dn'

        ewo_zero_dn = (
                (dataframe['EWO'] > dataframe['EWO_MEAN_DN']) &
                (dataframe['EWO'] < 0) &
                (dataframe['buy_decision'] > self.zero_dn.value) &
                (dataframe['gen_buy'] > self.b03.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_zero_dn, 'enter_long'] = 1
        dataframe.loc[ewo_zero_dn, 'enter_tag'] = 'ewo_zero_dn'

        ewo_fib_up = (
                (dataframe['EWO'] > dataframe['EWO_UP_FIB']) &
                (dataframe['buy_decision'] > self.fib_up.value) &
                (dataframe['gen_buy'] > self.b04.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_fib_up, 'enter_long'] = 1
        dataframe.loc[ewo_fib_up, 'enter_tag'] = 'ewo_fib_up'

        ewo_mean_up = (
                (dataframe['EWO'] < dataframe['EWO_UP_FIB']) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['buy_decision'] > self.mean_up.value) &
                (dataframe['gen_buy'] > self.b05.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_mean_up, 'enter_long'] = 1
        dataframe.loc[ewo_mean_up, 'enter_tag'] = 'ewo_mean_up'

        ewo_zero_up = (
                (dataframe['EWO'] < dataframe['EWO_MEAN_UP']) &
                (dataframe['EWO'] > 0) &
                (dataframe['buy_decision'] > self.zero_up.value) &
                (dataframe['gen_buy'] > self.b06.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_zero_dn, 'enter_long'] = 1
        dataframe.loc[ewo_zero_dn, 'enter_tag'] = 'ewo_zero_dn'

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ewo_fib_dns = (
                (dataframe['EWO'] < dataframe['EWO_DN_FIB']) &
                (dataframe['sell_decision'] > self.fib_dns.value) &
                (dataframe['gen_sell'] > self.s01.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_fib_dns, 'exit_long'] = 1
        dataframe.loc[ewo_fib_dns, 'exit_tag'] = 'ewo_fib_dns'

        ewo_mean_dns = (
                (dataframe['EWO'] > dataframe['EWO_DN_FIB']) &
                (dataframe['EWO'] < dataframe['EWO_MEAN_DN']) &
                (dataframe['sell_decision'] > self.mean_dns.value) &
                (dataframe['gen_sell'] > self.s02.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_mean_dns, 'exit_long'] = 1
        dataframe.loc[ewo_mean_dns, 'exit_tag'] = 'ewo_mean_dns'

        ewo_zero_dns = (
                (dataframe['EWO'] > dataframe['EWO_MEAN_DN']) &
                (dataframe['EWO'] < 0) &
                (dataframe['sell_decision'] > self.zero_dns.value) &
                (dataframe['gen_sell'] > self.s03.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_zero_dns, 'exit_long'] = 1
        dataframe.loc[ewo_zero_dns, 'exit_tag'] = 'ewo_zero_dns'

        ewo_fib_ups = (
                (dataframe['EWO'] > dataframe['EWO_UP_FIB']) &
                (dataframe['sell_decision'] > self.fib_ups.value) &
                (dataframe['gen_sell'] > self.s04.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_fib_ups, 'exit_long'] = 1
        dataframe.loc[ewo_fib_ups, 'exit_tag'] = 'ewo_fib_ups'

        ewo_mean_ups = (
                (dataframe['EWO'] < dataframe['EWO_UP_FIB']) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['sell_decision'] > self.mean_ups.value) &
                (dataframe['gen_sell'] > self.s05.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_mean_ups, 'exit_long'] = 1
        dataframe.loc[ewo_mean_ups, 'exit_tag'] = 'ewo_mean_ups'

        ewo_zero_ups = (
                (dataframe['EWO'] < dataframe['EWO_MEAN_UP']) &
                (dataframe['EWO'] > 0) &
                (dataframe['sell_decision'] > self.zero_ups.value) &
                (dataframe['gen_sell'] > self.s06.value) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[ewo_zero_dns, 'exit_long'] = 1
        dataframe.loc[ewo_zero_dns, 'exit_tag'] = 'ewo_zero_dns'

        return dataframe


def pct_change(a, b):
    return (b - a) / a




