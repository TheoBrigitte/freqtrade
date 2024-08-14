from skopt.space import Dimension, Integer

import logging
import logging
from datetime import datetime, timezone
from functools import reduce
from typing import List

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
import technical.indicators as ftt
from pandas import DataFrame, Series
from skopt.space import Dimension, Integer

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, DecimalParameter,
                                IntParameter, merge_informative_pair, stoploss_from_open)
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)

# ###############################################################################
# ###############################################################################
# @Farhad#0318 ( https://github.com/farfary/freqtrade_strategies )
#
# Based on buy signal from Al (alb#1349)
# ###############################################################################
# ###############################################################################

class MiniLambo(IStrategy):
    # Protection hyperspace params:
    protection_params = {
        "protection_cooldown_period": 2,
        "protection_maxdrawdown_lookback_period_candles": 35,
        "protection_maxdrawdown_max_allowed_drawdown": 0.097,
        "protection_maxdrawdown_stop_duration_candles": 1,
        "protection_maxdrawdown_trade_limit": 6,
        "protection_stoplossguard_lookback_period_candles": 16,
        "protection_stoplossguard_stop_duration_candles": 29,
        "protection_stoplossguard_trade_limit": 3,
    }

    protection_cooldown_period = IntParameter(low=1, high=48, default=protection_params['protection_cooldown_period'], space="protection", optimize=True)

    protection_maxdrawdown_lookback_period_candles = IntParameter(low=1, high=48, default=protection_params['protection_maxdrawdown_lookback_period_candles'], space="protection", optimize=True)
    protection_maxdrawdown_trade_limit = IntParameter(low=1, high=8, default=protection_params['protection_maxdrawdown_trade_limit'], space="protection", optimize=True)
    protection_maxdrawdown_stop_duration_candles = IntParameter(low=1, high=48, default=protection_params['protection_maxdrawdown_stop_duration_candles'], space="protection", optimize=True)
    protection_maxdrawdown_max_allowed_drawdown = DecimalParameter(low=0.01, high=0.20, default=protection_params['protection_maxdrawdown_max_allowed_drawdown'], space="protection", optimize=True)

    protection_stoplossguard_lookback_period_candles = IntParameter(low=1, high=48, default=protection_params['protection_stoplossguard_lookback_period_candles'], space="protection", optimize=True)
    protection_stoplossguard_trade_limit = IntParameter(low=1, high=8, default=protection_params['protection_stoplossguard_trade_limit'], space="protection", optimize=True)
    protection_stoplossguard_stop_duration_candles = IntParameter(low=1, high=48, default=protection_params['protection_stoplossguard_stop_duration_candles'], space="protection", optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.protection_cooldown_period.value
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.protection_maxdrawdown_lookback_period_candles.value,
                "trade_limit": self.protection_maxdrawdown_trade_limit.value,
                "stop_duration_candles": self.protection_maxdrawdown_stop_duration_candles.value,
                "max_allowed_drawdown": self.protection_maxdrawdown_max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.protection_stoplossguard_lookback_period_candles.value,
                "trade_limit": self.protection_stoplossguard_trade_limit.value,
                "stop_duration_candles": self.protection_stoplossguard_stop_duration_candles.value,
                "only_per_pair": False
            }
        ]

    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: dict):
            """
            Generate the ROI table that will be used by Hyperopt
            This implementation generates the default legacy Freqtrade ROI tables.
            Change it if you need different number of steps in the generated
            ROI tables or other structure of the ROI tables.
            Please keep it aligned with parameters in the 'roi' optimization
            hyperspace defined by the roi_space method.
            """
            roi_table = {}
            roi_table[0] = 0.05
            roi_table[params['roi_t6']] = 0.04
            roi_table[params['roi_t5']] = 0.03
            roi_table[params['roi_t4']] = 0.02
            roi_table[params['roi_t3']] = 0.01
            roi_table[params['roi_t2']] = 0.0001
            roi_table[params['roi_t1']] = -10

            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            """
            Values to search for each ROI steps
            Override it if you need some different ranges for the parameters in the
            'roi' optimization hyperspace.
            Please keep it aligned with the implementation of the
            generate_roi_table method.
            """
            return [
                Integer(240, 720, name='roi_t1'),
                Integer(120, 240, name='roi_t2'),
                Integer(90, 120, name='roi_t3'),
                Integer(60, 90, name='roi_t4'),
                Integer(30, 60, name='roi_t5'),
                Integer(1, 30, name='roi_t6'),
            ]

    # Buy hyperspace params:
    buy_params = {
        "lambo2_ema_14_factor": 1.053,
        "lambo2_pct_change_high_period": 80,
        "lambo2_pct_change_high_ratio": -0.175,
        "lambo2_pct_change_low_period": 60,
        "lambo2_pct_change_low_ratio": -0.05,
        "lambo2_rsi_14_limit": 54,
        "lambo2_rsi_21_limit": 43,
        "lambo2_rsi_4_limit": 53
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.158,
        "pPF_1": 0.012,
        "pPF_2": 0.079,
        "pSL_1": 0.009,
        "pSL_2": 0.064
    }

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "15": 0.04,
        "51": 0.03,
        "81": 0.02,
        "112": 0.01,
        "154": 0.0001,
        "400": -10
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = False

    timeframe = '1m'
    timeframe_info_15m = '15m'
    timeframe_info_1h = '1h'
    timeframe_info_4h = '4h'
    timeframe_info_1d = '1d'

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 200


    # hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True, optimize=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=True)


    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=False)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)
    lambo2_rsi_21_limit = IntParameter(5, 80, default=buy_params['lambo2_rsi_21_limit'], space='buy', optimize=True)

    lambo2_pct_change_low_period = IntParameter(1, 60, default=buy_params['lambo2_pct_change_low_period'], space='buy', optimize=True)
    lambo2_pct_change_low_ratio = DecimalParameter(low=-0.20, high=-0.01, decimals=3, default=buy_params['lambo2_pct_change_low_ratio'], space='buy', optimize=True)

    lambo2_pct_change_high_period = IntParameter(1, 180, default=buy_params['lambo2_pct_change_high_period'], space='buy', optimize=False)
    lambo2_pct_change_high_ratio = DecimalParameter(low=-0.30, high=-0.01, decimals=3, default=buy_params['lambo2_pct_change_high_ratio'], space='buy', optimize=False)


    # btc safe
    # btc_1h_rsi_21 = IntParameter(10, 90, default=buy_params['btc_1h_rsi_21'], space='buy', optimize=True)



    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
    #     sl_new = 1
    #
    #     if (current_profit > 0.2):
    #         sl_new = 0.05
    #     elif (current_profit > 0.1):
    #         sl_new = 0.03
    #     elif (current_profit > 0.06):
    #         sl_new = 0.02
    #     elif (current_profit > 0.03):
    #         sl_new = 0.015
    #     elif (current_profit > 0.015):
    #         sl_new = 0.0075
    #
    #     return sl_new

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [("BTC/USDT", "1d")]
        informative_pairs += [("BTC/USDT", "4h")]
        informative_pairs += [("BTC/USDT", "1h")]
        return informative_pairs

    def pop_df_btc_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_1d_df = self.dp.get_pair_dataframe("BTC/USDT", '1d')
        btc_1d_df['rsi_21'] = ta.RSI(btc_1d_df, timeperiod=21)
        return btc_1d_df

    def pop_df_btc_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_4h_df = self.dp.get_pair_dataframe("BTC/USDT", '4h')
        btc_4h_df['rsi_21'] = ta.RSI(btc_4h_df, timeperiod=21)
        return btc_4h_df

    def pop_df_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_1h_df = self.dp.get_pair_dataframe("BTC/USDT", '1h')
        return btc_1h_df

    def informative_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1d = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe_info_1d)
        return informative_1d

    def informative_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe_info_4h)
        return informative_4h

    def informative_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe_info_1h)
        informative_1h['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        return informative_1h

    def informative_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe_info_15m)
        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BTC 1D DataFrame
        # dataframe = merge_informative_pair(dataframe, self.pop_df_btc_1d(dataframe, metadata), self.timeframe, '1d', ffill=True)
        # # BTC 4H DataFrame
        # dataframe = merge_informative_pair(dataframe, self.pop_df_btc_4h(dataframe, metadata), self.timeframe, '4h', ffill=True)
        # # BTC 1H DataFrame
        # dataframe = merge_informative_pair(dataframe, self.pop_df_btc_1h(dataframe, metadata), self.timeframe, '1h', ffill=True)

        # Main Pair DataFrame
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        # informative_15m = self.informative_indicators_15m(dataframe, metadata)
        # informative_1h = self.informative_indicators_1h(dataframe, metadata)
        # informative_4h = self.informative_indicators_4h(dataframe, metadata)
        # informative_1d = self.informative_indicators_1d(dataframe, metadata)
        #
        # dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.timeframe_info_15m, ffill=True)
        # dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.timeframe_info_1h, ffill=True)
        # dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, self.timeframe_info_4h, ffill=True)
        # dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, self.timeframe_info_1d, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        lambo2 = (
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value))
                & (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value))
                & (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
                & (dataframe['rsi_21'] < int(self.lambo2_rsi_21_limit.value))
                & (dataframe['low'].pct_change(periods=self.lambo2_pct_change_low_period.value) < float(self.lambo2_pct_change_low_ratio.value))

                # & (dataframe['rsi_21'] > int(self.lambo2_rsi_21_limit.value))
                # & (dataframe['low'].pct_change(periods=self.lambo2_pct_change_high_period.value) > float(self.lambo2_pct_change_high_ratio.value))
                # & (dataframe['btc_rsi_21_1h'] > self.btc_1h_rsi_21.value)
        )
        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2 '
        conditions.append(lambo2)

        dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        trade.exit_reason = f'{exit_reason} ({trade.buy_tag})'
        return True

class MiniLambo_TBS(MiniLambo):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_exit tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = True

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]:
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price return None to stop
        # trailing buy (will start again at next buy signal) return 'forcebuy' to force immediate buy (example with
        # 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (
        # no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt = abs((last_candle['perc_norm']))
        default_offset = 0.004 * (1 + adapt)  # NOTE: default_offset 0.003 <--> 0.006
        # default_offset = adapt*0.01

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if (current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (
                trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (
                      current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)

        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if (len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1):
                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['buy_tag'] = last_candle['buy_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0

                            self.trailing_buy_info(pair, current_price)
                            logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(
                                    current_price * (1 + trailing_buy_offset),
                                    self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (
                                    trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (
                                    trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')

        return val

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:, 'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

# def bollinger_bands(stock_price, window_size, num_of_std):
#     rolling_mean = stock_price.rolling(window=window_size).mean()
#     rolling_std = stock_price.rolling(window=window_size).std()
#     lower_band = rolling_mean - (rolling_std * num_of_std)
#     return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)
#
#
# def ha_typical_price(bars):
#     res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
#     return Series(index=bars.index, data=res)
#
#
# def pct_change(a, b):
#     return (b - a) / a
