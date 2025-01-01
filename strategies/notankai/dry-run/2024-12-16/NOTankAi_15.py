
import warnings
warnings.filterwarnings('ignore')

import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    RealParameter,
    merge_informative_pair,
)
from scipy.signal import argrelextrema
import warnings
import math

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class NOTankAi_15(IStrategy):
    exit_profit_only = True
    trailing_stop = False
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    #startup_candle_count: int = 200
    stoploss = -0.99
    timeframe = "15m"

    # DCA
    position_adjustment_enable = True
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="entry", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="entry", optimize=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="entry", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2, default=1.4, decimals=1, space="entry", optimize=True, load=True
    )

    # Custom Functions
    increment = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="entry", optimize=True, load=True
    )
    last_entry_price = None

    # Protections
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {"color": "#f53580", "type": "line"},
                "&s-minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "&s-maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#ac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#ac7fc", "type": "line"},
            },
        },
    }

    @property
    def protections(self):
        prot = []
        prot.append(
            {"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}
        )
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24 * 3,
                    "trade_limit": 2,
                    "stop_duration_candles": self.stop_duration.value,
                    "only_per_pair": False,
                }
            )
        return prot

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return proposed_stake / self.max_dca_multiplier

    def custom_entry_price(
        self,
        pair: str,
        trade: Optional["Trade"],
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, last_updated = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        entry_price = (dataframe["close"].iat[-1] + dataframe["open"].iat[-1] + proposed_rate) / 3
        if proposed_rate < entry_price:
            entry_price = proposed_rate

        logger.info(
            f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}"
        )

        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0005:
            entry_price *= self.increment.value
            logger.info(
                f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}."
            )

        self.last_entry_price = entry_price

        return entry_price

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        if exit_reason == "partial_exit" and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f"{trade.pair} partial exit is below 0")
            return False
        if exit_reason == "trailing_stop_loss" and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f"{trade.pair} trailing stop price is below 0")
            return False
        return True

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)

        if current_profit > -0.15 and trade.nr_of_successful_entries == 1:
            return None
        if current_profit > -0.3 and trade.nr_of_successful_entries == 2:
            return None
        if current_profit > -0.6 and trade.nr_of_successful_entries == 3:
            return None

        try:
            stake_amount = filled_entries[0].cost
            if count_of_entries == 1:
                stake_amount = stake_amount * 1
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1
            else:
                stake_amount = stake_amount
            return stake_amount
        except Exception as exception:
            return None
        return None

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        window_size = 50
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        historical_close_prices = dataframe["close"].tail(window_size)
        historical_high_prices = dataframe["high"].tail(window_size)
        historical_low_prices = dataframe["low"].tail(window_size)
        base_leverage = 10

        rsi_values = ta.RSI(historical_close_prices, timeperiod=14)
        atr_values = ta.ATR(
            historical_high_prices, historical_low_prices, historical_close_prices, timeperiod=14
        )
        macd_line, signal_line, _ = ta.MACD(
            historical_close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        sma_values = ta.SMA(historical_close_prices, timeperiod=20)
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0.0
        current_macd = (
            macd_line[-1] - signal_line[-1] if len(macd_line) > 0 and len(signal_line) > 0 else 0.0
        )
        current_sma = sma_values[-1] if len(sma_values) > 0 else 0.0

        dynamic_rsi_low = (
            np.nanmin(rsi_values)
            if len(rsi_values) > 0 and not np.isnan(np.nanmin(rsi_values))
            else 30.0
        )
        dynamic_rsi_high = (
            np.nanmax(rsi_values)
            if len(rsi_values) > 0 and not np.isnan(np.nanmax(rsi_values))
            else 70.0
        )
        dynamic_atr_low = (
            np.nanmin(atr_values)
            if len(atr_values) > 0 and not np.isnan(np.nanmin(atr_values))
            else 0.002
        )
        dynamic_atr_high = (
            np.nanmax(atr_values)
            if len(atr_values) > 0 and not np.isnan(np.nanmax(atr_values))
            else 0.005
        )

        long_increase_factor = 1.5
        long_decrease_factor = 0.5
        short_increase_factor = 1.5
        short_decrease_factor = 0.5
        volatility_decrease_factor = 0.8

        if side == "long":
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase_factor
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            if current_macd > 0:
                base_leverage *= long_increase_factor
            if current_rate < current_sma:
                base_leverage *= long_decrease_factor

        elif side == "short":
            if current_rsi > dynamic_rsi_high:
                base_leverage *= short_increase_factor
            elif current_rsi < dynamic_rsi_low:
                base_leverage *= short_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            if current_macd < 0:
                base_leverage *= short_increase_factor
            if current_rate > current_sma:
                base_leverage *= short_decrease_factor

        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)

        return adjusted_leverage

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe["close"], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / dataframe["close"] * 1000
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["dpo"] = pta.dpo(dataframe["close"], length=40, centered=False)
        dataframe["%-dpo"] = dataframe["dpo"]
        dataframe["%-willr14"] = pta.willr(dataframe["high"], dataframe["low"], dataframe["close"])

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe["%-vwap_upperband"] = vwap_high
        dataframe["%-vwap_middleband"] = vwap
        dataframe["%-vwap_lowerband"] = vwap_low
        dataframe["%-vwap_width"] = (
            (dataframe["%-vwap_upperband"] - dataframe["%-vwap_lowerband"])
            / dataframe["%-vwap_middleband"]
        ) * 100
        dataframe = dataframe.copy()
        dataframe["%-dist_to_vwap_upperband"] = get_distance(
            dataframe["close"], dataframe["%-vwap_upperband"]
        )
        dataframe["%-dist_to_vwap_middleband"] = get_distance(
            dataframe["close"], dataframe["%-vwap_middleband"]
        )
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(
            dataframe["close"], dataframe["%-vwap_lowerband"]
        )
        dataframe["%-tail"] = (dataframe["close"] - dataframe["low"]).abs()
        dataframe["%-wick"] = (dataframe["high"] - dataframe["close"]).abs()
        dataframe["%-rawclose"] = dataframe["close"]
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["%-ha_open"] = heikinashi["open"]
        dataframe["%-ha_close"] = heikinashi["close"]
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["%-ha_high"] = heikinashi["high"]
        dataframe["%-ha_low"] = heikinashi["low"]
        dataframe["%-ha_closedelta"] = heikinashi["close"] - heikinashi["close"].shift()
        dataframe["%-ha_tail"] = heikinashi["close"] - heikinashi["low"]
        dataframe["%-ha_wick"] = heikinashi["high"] - heikinashi["close"]

        dataframe["%-HLC3"] = (heikinashi["high"] + heikinashi["low"] + heikinashi["close"]) / 3

        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value

        dataframe["%-+3/8"] = dataframe["[+3/8]P"]
        dataframe["%-+2/8"] = dataframe["[+2/8]P"]
        dataframe["%-+1/8"] = dataframe["[+1/8]P"]
        dataframe["%-8/8"] = dataframe["[8/8]P"]
        dataframe["%-7/8"] = dataframe["[7/8]P"]
        dataframe["%-6/8"] = dataframe["[6/8]P"]
        dataframe["%-5/8"] = dataframe["[5/8]P"]
        dataframe["%-4/8"] = dataframe["[4/8]P"]
        dataframe["%-3/8"] = dataframe["[3/8]P"]
        dataframe["%-2/8"] = dataframe["[2/8]P"]
        dataframe["%-1/8"] = dataframe["[1/8]P"]
        dataframe["%-0/8"] = dataframe["[0/8]P"]
        dataframe["%--1/8"] = dataframe["[-1/8]P"]
        dataframe["%--2/8"] = dataframe["[-2/8]P"]
        dataframe["%--3/8"] = dataframe["[-3/8]P"]

        dataframe["ema_2"] = ta.EMA(dataframe, timeperiod=2)
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[+3/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[+2/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[+1/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[8/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[4/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[0/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[-1/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[-2/8]P"])
        dataframe["%-distema2"] = get_distance(dataframe["ema_2"], dataframe["[-3/8]P"])

        dataframe["%-entrythreshold4"] = dataframe["%-tail"] - dataframe["[0/8]P"]
        dataframe["%-entrythreshold5"] = dataframe["%-tail"] - dataframe["[-1/8]P"]
        dataframe["%-entrythreshold6"] = dataframe["%-tail"] - dataframe["[-2/8]P"]
        dataframe["%-entrythreshold7"] = dataframe["%-tail"] - dataframe["[-3/8]P"]

        dataframe["%-exitthreshold4"] = dataframe["%-wick"] - dataframe["[8/8]P"]
        dataframe["%-exitthreshold5"] = dataframe["%-wick"] - dataframe["[+1/8]P"]
        dataframe["%-exitthreshold6"] = dataframe["%-wick"] - dataframe["[+2/8]P"]
        dataframe["%-exitthreshold7"] = dataframe["%-wick"] - dataframe["[+3/8]P"]

        dataframe["mmlextreme_oscillator"] = 100 * (
            (dataframe["close"] - dataframe["[-3/8]P"])
            / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"])
        )
        dataframe["%-mmlextreme_oscillator"] = dataframe["mmlextreme_oscillator"]

        dataframe["%-perc_change"] = (dataframe["high"] / dataframe["open"] - 1) * 100
        dataframe["%-candle_1perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x >= 1, 1, 0).sum())
            .shift()
        )
        dataframe["%-candle_2perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x >= 2, 1, 0).sum())
            .shift()
        )
        dataframe["%-candle_3perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x >= 3, 1, 0).sum())
            .shift()
        )
        dataframe["%-candle_5perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x >= 5, 1, 0).sum())
            .shift()
        )
        dataframe["%-candle_-1perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x <= -1, -1, 0).sum())
            .shift()
        )
        dataframe["%-candle_-2perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x <= -2, -1, 0).sum())
            .shift()
        )
        dataframe["%-candle_-3perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x <= -3, -1, 0).sum())
            .shift()
        )
        dataframe["%-candle_-5perc_50"] = (
            dataframe["%-perc_change"]
            .rolling(50)
            .apply(lambda x: np.where(x <= -5, -1, 0).sum())
            .shift()
        )
        dataframe["%-close_percentage"] = (dataframe["close"] - dataframe["low"]) / (
            dataframe["high"] - dataframe["low"]
        )
        dataframe["%-body_size"] = abs(dataframe["open"] - dataframe["close"])
        dataframe["%-range_size"] = dataframe["high"] - dataframe["low"]
        dataframe["%-body_range_ratio"] = dataframe["%-body_size"] / dataframe["%-range_size"]
        dataframe["%-upper_wick_size"] = dataframe["high"] - dataframe[["open", "close"]].max(
            axis=1
        )
        dataframe["%-upper_wick_range_ratio"] = (
            dataframe["%-upper_wick_size"] / dataframe["%-range_size"]
        )
        lookback_period = 10
        dataframe["%-max_high"] = dataframe["high"].rolling(50).max()
        dataframe["%-min_low"] = dataframe["low"].rolling(50).min()
        dataframe["%-close_position"] = (dataframe["close"] - dataframe["%-min_low"]) / (
            dataframe["%-max_high"] - dataframe["%-min_low"]
        )
        dataframe["%-current_candle_perc_change"] = (
            dataframe["high"] / dataframe["open"] - 1
        ) * 100
        dataframe["%-hi"] = ta.SMA(dataframe["high"], timeperiod=28)
        dataframe["%-lo"] = ta.SMA(dataframe["low"], timeperiod=28)
        dataframe["%-ema1"] = ta.EMA(dataframe["%-HLC3"], timeperiod=28)
        dataframe["%-ema2"] = ta.EMA(dataframe["%-ema1"], timeperiod=28)
        dataframe["%-d"] = dataframe["%-ema1"] - dataframe["%-ema2"]
        dataframe["%-mi"] = dataframe["%-ema1"] + dataframe["%-d"]
        dataframe["%-md"] = np.where(
            dataframe["%-mi"] > dataframe["%-hi"],
            dataframe["%-mi"] - dataframe["%-hi"],
            np.where(
                dataframe["%-mi"] < dataframe["%-lo"], dataframe["%-mi"] - dataframe["%-lo"], 0
            ),
        )
        dataframe["%-sb"] = ta.SMA(dataframe["%-md"], timeperiod=8)
        dataframe["%-sh"] = dataframe["%-md"] - dataframe["%-sb"]

        ap = 0.333 * (heikinashi["high"] + heikinashi["low"] + heikinashi["close"])
        dataframe["esa"] = ta.EMA(ap, timeperiod=9)
        dataframe["d"] = ta.EMA(abs(ap - dataframe["esa"]), timeperiod=9)
        dataframe["%-wave_ci"] = (ap - dataframe["esa"]) / (0.015 * dataframe["d"])
        dataframe["%-wave_t1"] = ta.EMA(dataframe["%-wave_ci"], timeperiod=12)
        dataframe["%-wave_t2"] = ta.SMA(dataframe["%-wave_t1"], timeperiod=4)
        dataframe["%-200sma"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["%-200sma_dist"] = get_distance(heikinashi["close"], dataframe["%-200sma"])

        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        maxima = np.zeros(len(dataframe))
        minima = np.zeros(len(dataframe))

        maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
        minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1

        dataframe["maxima"] = maxima
        dataframe["minima"] = minima

        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(dataframe["close"].values, np.less, order=5)[0]
        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=5)[0]
        dataframe.loc[min_peaks, "&s-extrema"] = -1
        dataframe.loc[max_peaks, "&s-extrema"] = 1

        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value

        dataframe["mmlextreme_oscillator"] = 100 * (
            (dataframe["close"] - dataframe["[4/8]P"])
            / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"])
        )
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()

        dataframe["min_threshold_mean"] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe["max_threshold_mean"] = dataframe["maxima_sort_threshold"].expanding().mean()

        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )

        pair = metadata["pair"]
        if dataframe["maxima"].iloc[-3] == 1 and dataframe["maxima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Maxima Detected - Potential Short!!!")
        if dataframe["minima"].iloc[-3] == 1 and dataframe["minima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Minima Detected - Potential Long!!!")

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Condições para entrada long
        df.loc[
            (
                (df["DI_catch"] == 1)  # Condição DI_catch
                & (df["maxima_check"] == 1)  # Condição maxima_check
                & (df["&s-extrema"] < 0)  # Condição extrema
                & (df["minima"].shift(1) == 1)  # Condição minima anterior
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] < 30)  # RSI abaixo de 30 (condição adicional para limitar entradas)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima")

        df.loc[
            (
                (df["minima_check"] == 0)  # Condição minima_check
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] < 30)  # RSI abaixo de 30 (condição adicional para limitar entradas)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Full Send")

        df.loc[
            (
                (df["DI_catch"] == 1)  # Condição DI_catch
                & (df["minima_check"] == 0)  # Condição minima_check
                & (df["minima_check"].shift(5) == 1)  # Condição minima_check anterior
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] < 30)  # RSI abaixo de 30 (condição adicional para limitar entradas)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Check")

        # Condições para entrada short
        df.loc[
            (
                (df["DI_catch"] == 1)  # Condição DI_catch
                & (df["minima_check"] == 1)  # Condição minima_check
                & (df["&s-extrema"] > 0)  # Condição extrema
                & (df["maxima"].shift(1) == 1)  # Condição maxima anterior
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] > 70)  # RSI acima de 70 (condição adicional para limitar entradas)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima")

        df.loc[
            (
                (df["maxima_check"] == 0)  # Condição maxima_check
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] > 70)  # RSI acima de 70 (condição adicional para limitar entradas)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Full Send")

        df.loc[
            (
                (df["DI_catch"] == 1)  # Condição DI_catch
                & (df["maxima_check"] == 0)  # Condição maxima_check
                & (df["maxima_check"].shift(5) == 1)  # Condição maxima_check anterior
                & (df["volume"] > 0)  # Volume maior que 0
                & (df["rsi"] > 70)  # RSI acima de 70 (condição adicional para limitar entradas)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Check")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[((df["maxima_check"] == 0) & (df["volume"] > 0)), ["exit_long", "exit_tag"]] = (
            1,
            "Maxima Check",
        )
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["&s-extrema"] > 0)
                & (df["maxima"].shift(1) == 1)
                & (df["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Maxima")
        df.loc[((df["maxima_check"] == 0) & (df["volume"] > 0)), ["exit_long", "exit_tag"]] = (
            1,
            "Maxima Full Send",
        )

        df.loc[((df["minima_check"] == 0) & (df["volume"] > 0)), ["exit_short", "exit_tag"]] = (
            1,
            "Minima Check",
        )
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["&s-extrema"] < 0)
                & (df["minima"].shift(1) == 1)
                & (df["volume"] > 0)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "Minima")
        df.loc[((df["minima_check"] == 0) & (df["volume"] > 0)), ["exit_short", "exit_tag"]] = (
            1,
            "Minima Full Send",
        )
        return df


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
    else:
        return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]


def chaikin_mf(df, periods=20):
    close = df["close"]
    low = df["low"]
    high = df["high"]
    volume = df["volume"]
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name="cmf")


def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df["vwap"] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df["vwap"].rolling(window=window_size).std()
    df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
    df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
    return df["vwap_low"], df["vwap"], df["vwap_high"]


def get_distance(p1, p2):
    return abs((p1) - (p2))


def calculate_murrey_math_levels(df, window_size=64):
    rolling_max_H = df["high"].rolling(window=window_size).max()
    rolling_min_L = df["low"].rolling(window=window_size).min()
    max_H = rolling_max_H
    min_L = rolling_min_L
    range_HL = max_H - min_L

    def calculate_fractal(v2):
        fractal = 0
        if 25000 < v2 <= 250000:
            fractal = 100000
        elif 2500 < v2 <= 25000:
            fractal = 10000
        elif 250 < v2 <= 2500:
            fractal = 1000
        elif 25 < v2 <= 250:
            fractal = 100
        elif 12.5 < v2 <= 25:
            fractal = 12.5
        elif 6.25 < v2 <= 12.5:
            fractal = 12.5
        elif 3.125 < v2 <= 6.25:
            fractal = 3.125
        elif 1.5625 < v2 <= 3.125:
            fractal = 3.125
        elif 0.390625 < v2 <= 1.5625:
            fractal = 1.5625
        elif 0 < v2 <= 0.390625:
            fractal = 0.1953125
        return fractal

    def calculate_octave(v1, v2, mn, mx):
        range_ = v2 - v1
        sum_ = np.floor(np.log(calculate_fractal(v1) / range_) / np.log(2))
        octave = calculate_fractal(v1) * (0.5**sum_)
        mn = np.floor(v1 / octave) * octave
        if mn + octave > v2:
            mx = mn + octave
        else:
            mx = mn + (2 * octave)
        return mx

    def calculate_x_values(v1, v2, mn, mx):
        dmml = (v2 - v1) / 8
        x_values = []
        midpoints = [mn + i * dmml for i in range(8)]
        for i in range(7):
            x_i = (midpoints[i] + midpoints[i + 1]) / 2
            x_values.append(x_i)
        finalH = max(x_values)
        return x_values, finalH

    def calculate_y_values(x_values, mn):
        y_values = []
        for x in x_values:
            if x > 0:
                y = mn
            else:
                y = 0
            y_values.append(y)
        return y_values

    def calculate_mml(mn, finalH, mx):
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (float([mx][0]) * 0.99875) + (dmml * 3)
        ml = []
        for i in range(0, 16):
            calc = mml - (dmml * (i))
            ml.append(calc)
        murrey_math_levels = {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0],
        }
        return mml, murrey_math_levels

    for i in range(len(df)):
        mn = np.min(min_L.iloc[: i + 1])
        mx = np.max(max_H.iloc[: i + 1])
        x_values, finalH = calculate_x_values(mn, mx, mn, mx)
        y_values = calculate_y_values(x_values, mn)
        finalL = np.min(y_values)
        mml, murrey_math_levels = calculate_mml(finalL, finalH, mx)
        for level, value in murrey_math_levels.items():
            df.at[df.index[i], level] = value

    return df


def PC(dataframe, in1, in2):
    df = dataframe.copy()

    pc = ((in2 - in1) / in1) * 100
    return pc
