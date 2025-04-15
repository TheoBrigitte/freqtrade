# --- Do not remove these libs ---
import random
from datetime import datetime
from datetime import timedelta
from functools import reduce

import numpy as np
# --------------------------------
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter
from freqtrade.strategy import DecimalParameter
from freqtrade.strategy import IntParameter
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy.interface import IStrategy


#   --------------------------------------------------------------------------------
#   Author: rextea      2021/05/29     Version: 5.0
#   --------------------------------------------------------------------------------
#   Strategy based on the legendary BinHV45:
#   https://github.com/freqtrade/freqtrade-strategies
#
#
#   Posted on Freqtrade discord channel: https://discord.gg/Xr4wUYc6

def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class BinMfiBTCv5003(IStrategy):
    timeframe = '5m'
    btc_timeframe = '5m'
    btc_pair = 'BTC/USDT'

    stoploss = -0.095
    use_custom_stoploss = True

    minimal_roi = {
        "0": 0.08,
        "1": 0.015,
        "10": 0.02,
        "90": 0.005
    }

    # protections = [
    #     {
    #         "method": "MaxDrawdown",
    #         "lookback_period": 280,
    #         "stop_duration": 180,
    #         "max_allowed_drawdown": 0.20
    #     }
    # ]

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    startup_candle_count: int = 100

    optimize_bear = True
    optimize_bull = True
    optimize_wild = True

    optimize_buy_1 = False
    optimize_buy_2 = False
    optimize_buy_3 = True

    optimize_stoploss = False
    optimize_ignore_roi = False
    optimize_protections = False

    btc_smadelta_buy = DecimalParameter(0.9, 1.10, default=1.03, space='buy', optimize=False)

    # --------------------------
    # Buy condition (#1) params:
    # --------------------------
    strict_bbdelta_close = DecimalParameter(0.0, 0.045, default=0.03344, space='buy',
                                            optimize=optimize_buy_1 and optimize_bear)
    strict_closedelta_close = DecimalParameter(-0.015, 0.045, default=0.00681, space='buy',
                                               optimize=optimize_buy_1 and optimize_bear)
    strict_tail_bbdelta = DecimalParameter(0.0, 2.0, default=1.73588, space='buy',
                                           optimize=optimize_buy_1 and optimize_bear)
    strict_mfi_limit = IntParameter(-15, 70, default=25, space='buy', optimize=optimize_buy_1 and optimize_bear)

    loose_bbdelta_close = DecimalParameter(0.0, 0.045, default=0.03344, space='buy',
                                           optimize=optimize_buy_1 and optimize_bull)
    loose_closedelta_close = DecimalParameter(-0.015, 0.045, default=0.00681, space='buy',
                                              optimize=optimize_buy_1 and optimize_bull)
    loose_tail_bbdelta = DecimalParameter(0.0, 2.0, default=1.73588, space='buy',
                                          optimize=optimize_buy_1 and optimize_bull)
    loose_mfi_limit = IntParameter(-15, 100, default=25, space='buy', optimize=optimize_buy_1 and optimize_bull)

    wild_bbdelta_close = DecimalParameter(0.0, 0.045, default=0.03344, space='buy',
                                          optimize=optimize_buy_1 and optimize_wild)
    wild_closedelta_close = DecimalParameter(-0.015, 0.045, default=0.00681, space='buy',
                                             optimize=optimize_buy_1 and optimize_wild)
    wild_tail_bbdelta = DecimalParameter(0.0, 2.0, default=1.73588, space='buy',
                                         optimize=optimize_buy_1 and optimize_wild)
    wild_mfi_limit = IntParameter(-15, 100, default=25, space='buy', optimize=optimize_buy_1 and optimize_wild)

    # --------------------------
    # Buy condition (#2) params:
    # --------------------------
    loose_ewo_bottom = DecimalParameter(-20.0, -8.0, default=-12.0, space='buy',
                                        optimize=optimize_buy_2 and optimize_bull)
    strict_ewo_bottom = DecimalParameter(-20.0, -8.0, default=-12.0, space='buy',
                                         optimize=optimize_buy_2 and optimize_bear)
    wild_ewo_bottom = DecimalParameter(-20.0, -8.0, default=-12.0, space='buy',
                                       optimize=optimize_buy_2 and optimize_wild)

    # --------------------------
    # Buy condition (#3) params:
    # --------------------------
    loose_ewo_bull = DecimalParameter(1.0, 20.0, default=6.0, space='buy', optimize=optimize_buy_3 and optimize_bull)
    loose_ewo_rsi = IntParameter(30, 70, default=55, space='buy', optimize=optimize_buy_3 and optimize_bull)

    strict_ewo_bull = DecimalParameter(1.0, 20.0, default=6.0, space='buy', optimize=optimize_buy_3 and optimize_bear)
    strict_ewo_rsi = IntParameter(30, 70, default=55, space='buy', optimize=optimize_buy_3 and optimize_bear)

    wild_ewo_bull = DecimalParameter(1.0, 20.0, default=6.0, space='buy', optimize=optimize_buy_3 and optimize_wild)
    wild_ewo_rsi = IntParameter(30, 70, default=55, space='buy', optimize=optimize_buy_3 and optimize_wild)

    # ----------------
    # StopLoss params:
    # ----------------
    btc_bail_pct = DecimalParameter(-0.1, 0, default=-0.018, space='sell', optimize=optimize_stoploss)
    btc_bail_pct_2 = DecimalParameter(-0.1, 0, default=-0.02, space='sell', optimize=optimize_stoploss)
    btc_bail_roc = DecimalParameter(-20.0, -1.0, default=-5.5, space='sell', optimize=optimize_stoploss)

    bail_after_period = CategoricalParameter([60, 120, 240, 280, 340, 400, 480, 600, 800, 1200], default=480,
                                             space='sell', optimize=optimize_stoploss)
    bail_ng_profit = CategoricalParameter([0, -0.01, -0.02, -0.03, -0.05, -0.06], default=-0.03,
                                          space='sell', optimize=optimize_stoploss)
    bail_ng_profit_2 = CategoricalParameter([-0.05, -0.06, -0.07, -0.08, -0.09], default=-0.07,
                                            space='sell', optimize=optimize_stoploss)
    bail_ewo = CategoricalParameter([0, -0.5, -1, -1.5, -2, -2.5], default=-1,
                                    space='sell', optimize=optimize_stoploss)
    bail_ewo_2 = CategoricalParameter([0, -0.5, -1, -1.5, -2, -2.5], default=0,
                                      space='sell', optimize=optimize_stoploss)

    # ------------------
    # Ignore ROI params:
    # ------------------
    ignore_roi_ewo = DecimalParameter(0.0, 10.0, default=3.0, space='sell', optimize=optimize_ignore_roi)
    ignore_roi_rsi = IntParameter(30, 90, default=50, space='sell', optimize=optimize_ignore_roi)

    # -------------------
    # Protections params:
    # -------------------
    btc_minus_pct = DecimalParameter(-0.1, 0, default=-0.004, space='buy', optimize=optimize_protections)
    btc_plus_pct = DecimalParameter(-0.01, 0.05, default=-0.005, space='buy', optimize=optimize_protections)
    btc_low_rsi = IntParameter(10, 70, default=35, space='buy', optimize=optimize_protections)
    btc_sma_ratio = DecimalParameter(1.0, 1.05, default=1.01, space='buy', optimize=optimize_protections)

    # Buy hyperspace params:
    buy_params = {
        "loose_ewo_bull": 16.831,
        "loose_ewo_rsi": 38,
        "strict_ewo_bull": 4.377,
        "strict_ewo_rsi": 35,
        "wild_ewo_bull": 19.547,
        "wild_ewo_rsi": 51,
        "btc_low_rsi": 35,  # value loaded from strategy
        "btc_minus_pct": -0.004,  # value loaded from strategy
        "btc_plus_pct": -0.005,  # value loaded from strategy
        "btc_sma_ratio": 1.01,  # value loaded from strategy
        "btc_smadelta_buy": 1.03,  # value loaded from strategy
        "loose_bbdelta_close": 0.041,  # value loaded from strategy
        "loose_closedelta_close": 0.044,  # value loaded from strategy
        "loose_ewo_bottom": -9.059,  # value loaded from strategy
        "loose_mfi_limit": 75,  # value loaded from strategy
        "loose_tail_bbdelta": 0.099,  # value loaded from strategy
        "strict_bbdelta_close": 0.012,  # value loaded from strategy
        "strict_closedelta_close": 0.042,  # value loaded from strategy
        "strict_ewo_bottom": -18.952,  # value loaded from strategy
        "strict_mfi_limit": 38,  # value loaded from strategy
        "strict_tail_bbdelta": 0.103,  # value loaded from strategy
        "wild_bbdelta_close": 0.012,  # value loaded from strategy
        "wild_closedelta_close": 0.036,  # value loaded from strategy
        "wild_ewo_bottom": -8.294,  # value loaded from strategy
        "wild_mfi_limit": 58,  # value loaded from strategy
        "wild_tail_bbdelta": 1.081,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "ignore_roi_ewo": 2.989,
        "ignore_roi_rsi": 57,
        "bail_after_period": 240,  # value loaded from strategy
        "bail_ewo": -0.5,  # value loaded from strategy
        "bail_ewo_2": -2.5,  # value loaded from strategy
        "bail_ng_profit": -0.06,  # value loaded from strategy
        "bail_ng_profit_2": -0.08,  # value loaded from strategy
        "btc_bail_pct": -0.024,  # value loaded from strategy
        "btc_bail_pct_2": -0.091,  # value loaded from strategy
        "btc_bail_roc": -1.824,  # value loaded from strategy
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs.append((self.btc_pair, self.btc_timeframe))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---------------
        # BTC indicators:
        # ---------------
        informative = self.dp.get_pair_dataframe(self.btc_pair, self.btc_timeframe)
        informative['btc-open'] = informative['open']
        informative['btc-low'] = informative['low']
        informative['btc-close'] = informative['close']
        informative['btc-volume'] = informative['volume']
        informative['btc-sma25'] = ta.SMA(informative, timeperiod=25)
        informative['btc-sma50'] = ta.SMA(informative, timeperiod=50)
        informative['btc-sma100'] = ta.SMA(informative, timeperiod=100)
        informative['btc-bull'] = informative['btc-sma25'].lt(informative['close'])
        informative['btc-bear'] = informative['btc-sma25'].gt(informative['close'])
        informative['btc-roc'] = ta.ROC(informative, timeperiod=6)
        informative['btc-rsi'] = ta.RSI(informative, timeperiod=14)
        informative['btc-mfi'] = ta.MFI(informative, timeperiod=14)
        informative['btc-pct-change'] = informative['close'].pct_change()
        informative['btc-plus-di'] = ta.PLUS_DI(informative)
        informative['btc-minus-di'] = ta.MINUS_DI(informative)
        informative['btc-red-candle'] = np.where(informative['btc-close'] < informative['btc-close'].shift(), 1, 0)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.btc_timeframe, ffill=True)
        skip_columns = [(s + "_" + self.btc_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.btc_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        # ----------------
        # Coin Indicators:
        # ----------------
        dataframe['hl2'] = (dataframe["high"] + dataframe["low"]) / 2
        mid, lower = bollinger_bands(dataframe['hl2'], window_size=16, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['EWO'] = EWO(dataframe, 5, 35)
        dataframe['3close'] = dataframe['close'].rolling(window=3).mean()
        dataframe['pct-change'] = dataframe['close'].pct_change()

        dataframe['sma-25'] = ta.SMA(dataframe, timeperiod=25)
        dataframe['sma-50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['coin-bull'] = dataframe['sma-25'].lt(dataframe['close'])
        dataframe['coin-bear'] = dataframe['sma-50'].gt(dataframe['close'])

        dataframe['4h-pct-change'] = dataframe['close'].pct_change(48)

        # ---------------
        # 1h indicators:
        # ---------------
        informative = self.dp.get_pair_dataframe(metadata['pair'], '1h')
        informative['1h-rsi'] = ta.RSI(informative, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, '1h', ffill=True)
        skip_columns = [(s + "_1h") for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_1h", "") if (not s in skip_columns) else s,
            inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---------------
        # Buy conditions:
        # ---------------
        buy_conditions = []

        # WILD MARKET CONDITIONS:
        buy_conditions.append(
            (dataframe['btc-close'] > dataframe['btc-sma100']) &
            (
                    (dataframe['mfi'] <= self.wild_mfi_limit.value) &
                    (
                            dataframe['lower'].shift().gt(0) &
                            dataframe['bbdelta'].gt(dataframe['close'] * self.wild_bbdelta_close.value) &
                            dataframe['closedelta'].gt(dataframe['close'] * self.wild_closedelta_close.value) &
                            dataframe['tail'].lt(dataframe['bbdelta'] * self.wild_tail_bbdelta.value) &
                            dataframe['close'].lt(dataframe['lower'].shift()) &
                            dataframe['close'].le(dataframe['close'].shift())
                    )
                    |
                    # Elliot wave trend: buy when it's really low and rsi started ascending
                    (
                            (dataframe['EWO'] < self.wild_ewo_bottom.value) &
                            (dataframe['rsi'] > dataframe['rsi'].shift())
                    )
                    |
                    # Buy when Elliot wave trend is up, coin sma is up, and rsi not too high:
                    (
                            (dataframe['coin-bull']) &
                            (dataframe['EWO'] > self.wild_ewo_bull.value) &
                            (dataframe['EWO'].rolling(16).min() < 0) &
                            (dataframe['1h-rsi'] < self.wild_ewo_rsi.value)
                    )
            )
        )

        # BULL MARKET CONDITIONS:
        buy_conditions.append(
            (dataframe['btc-close'] > dataframe['btc-sma50'] * self.btc_smadelta_buy.value) &
            (
                    (dataframe['mfi'] <= self.loose_mfi_limit.value) &
                    (
                            dataframe['lower'].shift().gt(0) &
                            dataframe['bbdelta'].gt(dataframe['close'] * self.loose_bbdelta_close.value) &
                            dataframe['closedelta'].gt(dataframe['close'] * self.loose_closedelta_close.value) &
                            dataframe['tail'].lt(dataframe['bbdelta'] * self.loose_tail_bbdelta.value) &
                            dataframe['close'].lt(dataframe['lower'].shift()) &
                            dataframe['close'].le(dataframe['close'].shift())
                    )
                    |
                    # Elliot wave trend: buy when it's really low and rsi started ascending
                    (
                            (dataframe['EWO'] < self.loose_ewo_bottom.value) &
                            (dataframe['rsi'] > dataframe['rsi'].shift())
                    )
                    |
                    # Buy when Elliot wave trend is up, coin sma is up, and rsi not too high:
                    (
                            (dataframe['coin-bull']) &
                            (dataframe['EWO'] > self.loose_ewo_bull.value) &
                            (dataframe['EWO'].rolling(16).min() < 0) &
                            (dataframe['1h-rsi'] < self.loose_ewo_rsi.value)
                    )
            )
        )

        # BEAR MARKET CONDITIONS:
        buy_conditions.append(
            (dataframe['btc-close'] < dataframe['btc-sma50'] * self.btc_smadelta_buy.value) &
            (
                    (dataframe['mfi'] <= self.strict_mfi_limit.value) &
                    (
                            dataframe['lower'].shift().gt(0) &
                            dataframe['bbdelta'].gt(dataframe['close'] * self.strict_bbdelta_close.value) &
                            dataframe['closedelta'].gt(dataframe['close'] * self.strict_closedelta_close.value) &
                            dataframe['tail'].lt(dataframe['bbdelta'] * self.strict_tail_bbdelta.value) &
                            dataframe['close'].lt(dataframe['lower'].shift()) &
                            dataframe['close'].le(dataframe['close'].shift())
                    )
                    |
                    # Elliot wave trend: buy when it's really low and rsi started ascending
                    (
                            (dataframe['EWO'] < self.strict_ewo_bottom.value) &
                            (dataframe['rsi'] > dataframe['rsi'].shift())
                    )
                    |
                    # Buy when Elliot wave trend is up, coin sma is up, and rsi not too high:
                    (
                            (dataframe['coin-bull']) &
                            (dataframe['EWO'] > self.strict_ewo_bull.value) &
                            (dataframe['EWO'].rolling(16).min() < 0) &
                            (dataframe['1h-rsi'] < self.strict_ewo_rsi.value)
                    )
            )
        )

        if buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, buy_conditions), 'buy'] = 1

        # dataframe.loc[:, 'buy'] = 1

        # ------------
        # Protections:
        # ------------
        protections = []

        protections.append(
            (dataframe['pct-change'].rolling(2).sum() < -0.063)
        )

        protections.append(
            (dataframe['4h-pct-change'].rolling(3).max() > 0.225) &
            (dataframe['pct-change'] < 0) &
            (dataframe['rsi'] > 53)
        )

        protections.append(
            (dataframe['open'].shift(1) > dataframe['close'].shift(1)) &
            (dataframe['open'].shift(2) > dataframe['close'].shift(2)) &
            (dataframe['open'].shift(3) > dataframe['close'].shift(3)) &
            (dataframe['open'].shift(4) > dataframe['close'].shift(4)) &
            (dataframe['volume'] > dataframe['volume'].shift(1))
        )

        protections.append(
            ((dataframe['btc-close'] * self.btc_sma_ratio.value) < dataframe[f'btc-sma25']) &
            (dataframe['btc-rsi'] <= self.btc_low_rsi.value) &
            (dataframe['btc-pct-change'].rolling(3).sum() < self.btc_minus_pct.value) &
            (dataframe['btc-pct-change'] < self.btc_plus_pct.value)
        )

        protections.append(
            (dataframe['btc-open'].shift(1) > dataframe['btc-close'].shift(1)) &
            (dataframe['btc-open'].shift(2) > dataframe['btc-close'].shift(2)) &
            (dataframe['btc-open'].shift(3) > dataframe['btc-close'].shift(3)) &
            (dataframe['btc-open'].shift(4) > dataframe['btc-close'].shift(4)) &
            (dataframe['btc-volume'] > dataframe['btc-volume'].shift(1))
        )

        if protections:
            dataframe.loc[reduce(lambda x, y: x | y, protections), 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # immediately bailout when BTC dumping:
        if current_candle['btc-pct-change'] < self.btc_bail_pct.value or \
                current_candle['btc-roc'] < self.btc_bail_roc.value or \
                (current_candle['btc-pct-change'] < self.btc_bail_pct_2.value and current_candle['btc-bear']):
            return -0.01

        # Manage losing trades:
        if current_profit < 0 and (current_time - timedelta(minutes=10) > trade.open_date_utc):
            if current_candle['coin-bear'] and current_candle['pct-change'] < -0.033:
                return -0.01
            # if current_candle['EWO'] < -4 and current_candle['EWO'] < before_candle['EWO']:
            #     return -0.01
            if current_profit > self.bail_ng_profit.value and \
                    current_time - timedelta(minutes=int(self.bail_after_period.value)) > trade.open_date_utc and \
                    current_candle['EWO'] < self.bail_ewo.value:
                return -0.01
            if current_profit < self.bail_ng_profit_2.value and \
                    current_candle['EWO'] < self.bail_ewo_2.value:
                return -0.01

        return 0.99

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, current_time, **kwargs) -> bool:
        # if trade.open_date_utc == current_time:
        #     return False

        # Ignore ROI if seems to go up:
        if sell_reason == 'roi':
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            if last_candle['rsi'] > self.ignore_roi_rsi.value or \
                    last_candle['EWO'] > self.ignore_roi_ewo.value or \
                    last_candle['close'] > last_candle['3close']:
                return False
        return True

    # Shuffle that backtest bro
    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, **kwargs) -> bool:
    #     rand = random.randint(1, 10)
    #     if rand == 1:
    #         return False
    #
    #     return True
