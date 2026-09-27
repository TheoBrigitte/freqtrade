import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
    BooleanParameter,
)
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade


###########################################################################################################
##				NostalgiaForInfinityV3 by iterativ, modded by stash										 ##
##																									   ##
##	Strategy for Freqtrade https://github.com/freqtrade/freqtrade									  ##
##																									   ##
###########################################################################################################


class NFI3(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.01,
    }

    stoploss = -0.99  # effectively disabled.

    timeframe = "15m"
    inf_1h = "30m"

    process_only_new_candles = True

    startup_candle_count = 999

    #############################################################

    buy_params = {
        #############
        # Enable/Disable conditions
        "entry_condition_1_enable": True,
        "entry_condition_2_enable": True,
        "entry_condition_3_enable": True,
        "entry_condition_4_enable": True,
        "entry_condition_5_enable": True,
        "entry_condition_6_enable": True,
        "entry_condition_7_enable": True,
        "entry_condition_8_enable": True,
        "entry_condition_9_enable": True,
        "entry_condition_10_enable": True,
    }

    sell_params = {
        #############
        # Enable/Disable conditions
        "exit_condition_1_enable": True,
        "exit_condition_2_enable": True,
        "exit_condition_3_enable": True,
        "exit_condition_4_enable": True,
        "exit_condition_5_enable": True,
        "exit_condition_6_enable": True,
        "exit_condition_7_enable": True,
        "exit_condition_8_enable": True,
    }

    ############################################################################

    # entry

    entry_condition_1_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_2_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_3_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_4_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_5_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_6_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_7_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_8_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_9_enable = BooleanParameter(default=True, space="buy", optimize=False)
    entry_condition_10_enable = BooleanParameter(
        default=True, space="buy", optimize=False
    )

    entry_dip_threshold_0 = DecimalParameter(
        0.001, 0.1, default=0.03, space="buy", decimals=3, optimize=False
    )
    entry_dip_threshold_1 = DecimalParameter(
        0.001, 0.2, default=0.12, space="buy", decimals=3, optimize=False
    )
    entry_dip_threshold_2 = DecimalParameter(
        0.05, 0.4, default=0.3, space="buy", decimals=3, optimize=False
    )
    entry_dip_threshold_3 = DecimalParameter(
        0.2, 0.5, default=0.4, space="buy", decimals=3, optimize=False
    )

    optimize_entry_1 = False
    entry_volume_1 = DecimalParameter(
        1.0, 30.0, default=2.0, space="buy", decimals=1, optimize=optimize_entry_1
    )
    entry_min_inc_1 = DecimalParameter(
        0.005, 0.05, default=0.029, space="buy", decimals=3, optimize=optimize_entry_1
    )
    entry_rsi_1h_min_1 = DecimalParameter(
        40.0, 70.0, default=45.25, space="buy", decimals=2, optimize=optimize_entry_1
    )
    entry_rsi_1h_max_1 = DecimalParameter(
        70.0, 90.0, default=85.06, space="buy", decimals=2, optimize=optimize_entry_1
    )
    entry_rsi_1 = DecimalParameter(
        30.0, 40.0, default=36.64, space="buy", decimals=2, optimize=optimize_entry_1
    )
    entry_mfi_1 = DecimalParameter(
        36.0, 65.0, default=45.25, space="buy", decimals=2, optimize=optimize_entry_1
    )

    optimize_entry_2 = False
    entry_volume_2 = DecimalParameter(
        1.0, 10.0, default=2.96, space="buy", decimals=2, optimize=optimize_entry_2
    )
    entry_ema_relative_2 = DecimalParameter(
        0.005, 0.08, default=0.006, space="buy", decimals=3, optimize=optimize_entry_2
    )
    entry_rsi_1h_min_2 = DecimalParameter(
        40.0, 70.0, default=63.91, space="buy", decimals=2, optimize=optimize_entry_2
    )
    entry_rsi_1h_max_2 = DecimalParameter(
        70.0, 95.0, default=89.94, space="buy", decimals=2, optimize=optimize_entry_2
    )
    entry_rsi_1h_diff_2 = DecimalParameter(
        35.0, 55.0, default=38.69, space="buy", decimals=2, optimize=optimize_entry_2
    )
    entry_mfi_2 = DecimalParameter(
        36.0, 65.0, default=53.89, space="buy", decimals=2, optimize=optimize_entry_2
    )

    entry_bb40_bbdelta_close = DecimalParameter(
        0.005, 0.06, default=0.057, space="buy", optimize=False
    )
    entry_bb40_closedelta_close = DecimalParameter(
        0.01, 0.03, default=0.023, space="buy", optimize=False
    )
    entry_bb40_tail_bbdelta = DecimalParameter(
        0.15, 0.45, default=0.418, space="buy", optimize=False
    )

    entry_bb20_close_bblowerband = DecimalParameter(
        0.7, 1.1, default=0.98, space="buy", optimize=False
    )
    entry_bb20_volume = IntParameter(18, 35, default=24, space="buy", optimize=False)

    entry_volume_5 = DecimalParameter(
        1.0, 10.0, default=4.12, space="buy", decimals=2, optimize=False
    )
    entry_ema_open_mult_5 = DecimalParameter(
        0.01, 0.04, default=0.019, space="buy", decimals=3, optimize=False
    )

    entry_volume_6 = DecimalParameter(
        1.0, 10.0, default=1.48, space="buy", decimals=2, optimize=False
    )
    entry_ema_open_mult_6 = DecimalParameter(
        0.025, 0.05, default=0.033, space="buy", decimals=3, optimize=False
    )

    entry_volume_7 = DecimalParameter(
        1.0, 10.0, default=7.04, space="buy", decimals=2, optimize=False
    )
    entry_ema_open_mult_7 = DecimalParameter(
        0.015, 0.03, default=0.02, space="buy", decimals=3, optimize=False
    )
    entry_rsi_7 = DecimalParameter(
        24.0, 50.0, default=41.09, space="buy", decimals=2, optimize=False
    )

    entry_rsi_8 = DecimalParameter(
        30.0, 50.0, default=46.0, space="buy", decimals=1, optimize=False
    )

    entry_volume_9 = DecimalParameter(
        1.0, 30.0, default=17.0, space="buy", decimals=1, optimize=False
    )
    entry_bb_offset_9 = DecimalParameter(
        0.97, 1.05, default=0.98, space="buy", decimals=3, optimize=False
    )

    entry_volume_10 = DecimalParameter(
        1.0, 26.0, default=9.6, space="buy", decimals=1, optimize=False
    )
    entry_bb_offset_10 = DecimalParameter(
        0.97, 1.05, default=0.994, space="buy", decimals=3, optimize=False
    )
    entry_rsi_1h_10 = DecimalParameter(
        15.0, 40.0, default=30.2, space="buy", decimals=1, optimize=False
    )

    # exit

    exit_condition_1_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_2_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_3_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_4_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_5_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_6_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_7_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )
    exit_condition_8_enable = BooleanParameter(
        default=True, space="sell", optimize=False
    )

    exit_rsi_bb_1 = DecimalParameter(
        60.0, 80.0, default=79.5, space="sell", decimals=1, optimize=False
    )

    exit_rsi_bb_2 = DecimalParameter(
        72.0, 90.0, default=81, space="sell", decimals=1, optimize=False
    )

    exit_rsi_main_3 = DecimalParameter(
        77.0, 90.0, default=82, space="sell", decimals=1, optimize=False
    )

    exit_dual_rsi_rsi_4 = DecimalParameter(
        72.0, 84.0, default=73.4, space="sell", decimals=1, optimize=False
    )
    exit_dual_rsi_rsi_1h_4 = DecimalParameter(
        78.0, 92.0, default=79.6, space="sell", decimals=1, optimize=False
    )

    exit_ema_relative_5 = DecimalParameter(
        0.005, 0.05, default=0.024, space="sell", optimize=False
    )
    exit_rsi_diff_5 = DecimalParameter(
        0.0, 20.0, default=4.382, space="sell", optimize=False
    )

    exit_rsi_under_6 = DecimalParameter(
        72.0, 90.0, default=87.708, space="sell", decimals=1, optimize=False
    )

    exit_rsi_1h_7 = DecimalParameter(
        80.0, 95.0, default=81.7, space="sell", decimals=1, optimize=False
    )

    exit_bb_relative_8 = DecimalParameter(
        1.05, 1.3, default=1.1, space="sell", decimals=3, optimize=False
    )

    exit_custom_profit_1 = DecimalParameter(
        0.01, 0.20, default=0.01, space="sell", decimals=2, optimize=False
    )
    exit_custom_rsi_1 = DecimalParameter(
        30.0, 50.0, default=38.65, space="sell", decimals=2, optimize=False
    )
    exit_custom_profit_2 = DecimalParameter(
        0.01, 0.20, default=0.05, space="sell", decimals=2, optimize=False
    )
    exit_custom_rsi_2 = DecimalParameter(
        34.0, 50.0, default=43.37, space="sell", decimals=2, optimize=False
    )
    exit_custom_profit_3 = DecimalParameter(
        0.15, 0.30, default=0.25, space="sell", decimals=2, optimize=False
    )
    exit_custom_rsi_3 = DecimalParameter(
        38.0, 55.0, default=51.87, space="sell", decimals=2, optimize=False
    )
    exit_custom_profit_4 = DecimalParameter(
        0.3, 0.7, default=0.45, space="sell", decimals=2, optimize=False
    )
    exit_custom_rsi_4 = DecimalParameter(
        40.0, 58.0, default=50.35, space="sell", decimals=2, optimize=False
    )

    exit_custom_under_profit_1 = DecimalParameter(
        0.01, 0.10, default=0.02, space="sell", decimals=3, optimize=False
    )
    exit_custom_under_profit_2 = DecimalParameter(
        0.01, 0.10, default=0.025, space="sell", decimals=3, optimize=False
    )
    exit_custom_under_profit_3 = DecimalParameter(
        0.05, 0.3, default=0.07, space="sell", decimals=3, optimize=False
    )

    exit_trail_profit_min_1 = DecimalParameter(
        0.1, 0.25, default=0.166, space="sell", decimals=3, optimize=False
    )
    exit_trail_profit_max_1 = DecimalParameter(
        0.3, 0.5, default=0.38, space="sell", decimals=2, optimize=False
    )
    exit_trail_down_1 = DecimalParameter(
        0.04, 0.2, default=0.154, space="sell", decimals=3, optimize=False
    )

    exit_trail_profit_min_2 = DecimalParameter(
        0.01, 0.1, default=0.035, space="sell", decimals=3, optimize=False
    )
    exit_trail_profit_max_2 = DecimalParameter(
        0.08, 0.25, default=0.1, space="sell", decimals=2, optimize=False
    )
    exit_trail_down_2 = DecimalParameter(
        0.04, 0.2, default=0.045, space="sell", decimals=3, optimize=False
    )

    ############################################################################

    @informative(inf_1h)
    def populate_indicators_inf(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        # EMA
        dataframe["ema_15"] = ta.EMA(dataframe, timeperiod=15)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        # SMA
        dataframe["sma_50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        # BB
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        drop_columns = ["open", "high", "low", "close", "volume"]
        dataframe.drop(
            columns=dataframe.columns.intersection(drop_columns), inplace=True
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # BB 40
        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()
        dataframe["closedelta"] = (
            dataframe["close"] - dataframe["close"].shift()
        ).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()

        # BB 20
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(window=30).mean()

        # EMA
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe["sma_5"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

        dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Alligator
        dataframe["lips"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["smma_lips"] = dataframe["lips"].rolling(3).mean()
        dataframe["teeth"] = ta.SMA(dataframe, timeperiod=8)
        dataframe["smma_teeth"] = dataframe["teeth"].rolling(5).mean()
        dataframe["jaw"] = ta.SMA(dataframe, timeperiod=13)
        dataframe["smma_jaw"] = dataframe["jaw"].rolling(8).mean()

        # Volume
        dataframe["volume_mean_4"] = dataframe["volume"].rolling(4).mean().shift(1)

        # If don't exceed the dip limits
        dataframe["safe_dips"] = (
            (
                ((dataframe["open"] - dataframe["close"]) / dataframe["close"])
                < self.entry_dip_threshold_0.value
            )
            & (
                (
                    (dataframe["open"].rolling(2).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.entry_dip_threshold_1.value
            )
            & (
                (
                    (dataframe["open"].rolling(12).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.entry_dip_threshold_2.value
            )
            & (
                (
                    (dataframe["open"].rolling(144).max() - dataframe["close"])
                    / dataframe["close"]
                )
                < self.entry_dip_threshold_3.value
            )
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.entry_condition_1_enable.value
                & (dataframe["ema_50_30m"] > dataframe["ema_200_30m"])
                & (dataframe["sma_200"] > dataframe["sma_200"].shift(20))
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_1.value
                    > dataframe["volume"]
                )
                & (
                    (
                        (dataframe["close"] - dataframe["open"].rolling(36).min())
                        / dataframe["open"].rolling(36).min()
                    )
                    > self.entry_min_inc_1.value
                )
                & (dataframe["rsi_30m"] > self.entry_rsi_1h_min_1.value)
                & (dataframe["rsi_30m"] < self.entry_rsi_1h_max_1.value)
                & (dataframe["rsi"] < self.entry_rsi_1.value)
                & (dataframe["mfi"] < self.entry_mfi_1.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_2_enable.value
                & (dataframe["close"] < dataframe["sma_5"])
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_50"] > dataframe["ema_100"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_2.value
                    > dataframe["volume"]
                )
                & (
                    ((dataframe["close"] - dataframe["ema_200"]) / dataframe["ema_200"])
                    < self.entry_ema_relative_2.value
                )
                & (dataframe["rsi_30m"] > self.entry_rsi_1h_min_2.value)
                & (dataframe["rsi_30m"] < self.entry_rsi_1h_max_2.value)
                & (
                    dataframe["rsi"]
                    < dataframe["rsi_30m"] - self.entry_rsi_1h_diff_2.value
                )
                & (dataframe["mfi"] < self.entry_mfi_2.value)
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_3_enable.value
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["safe_dips"])
                & dataframe["lower"].shift().gt(0)
                & dataframe["bbdelta"].gt(
                    dataframe["close"] * self.entry_bb40_bbdelta_close.value
                )
                & dataframe["closedelta"].gt(
                    dataframe["close"] * self.entry_bb40_closedelta_close.value
                )
                & dataframe["tail"].lt(
                    dataframe["bbdelta"] * self.entry_bb40_tail_bbdelta.value
                )
                & dataframe["close"].lt(dataframe["lower"].shift())
                & dataframe["close"].le(dataframe["close"].shift())
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_4_enable.value
                & (dataframe["close"] > dataframe["ema_100"])
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_50"] > dataframe["ema_100"])
                & (dataframe["ema_15_30m"] > dataframe["ema_50_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_200_30m"])
                & (
                    (
                        (dataframe["open"].rolling(2).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.entry_dip_threshold_1.value
                )
                & (
                    (
                        (dataframe["open"].rolling(12).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.entry_dip_threshold_2.value
                )
                & (
                    (
                        (dataframe["open"].rolling(144).max() - dataframe["close"])
                        / dataframe["close"]
                    )
                    < self.entry_dip_threshold_3.value
                )
                & (dataframe["close"] < dataframe["ema_50"])
                & (
                    dataframe["close"]
                    < self.entry_bb20_close_bblowerband.value * dataframe["bb_lowerband"]
                )
                & (
                    dataframe["volume"]
                    < (
                        dataframe["volume_mean_30"].shift(1)
                        * self.entry_bb20_volume.value
                    )
                )
            )
        )

        conditions.append(
            (
                self.entry_condition_5_enable.value
                &
                # (dataframe['close'] > dataframe['ema_200']) &
                (dataframe["close"] > dataframe["ema_100_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_5.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.entry_ema_open_mult_5.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_6_enable.value
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_6.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.entry_ema_open_mult_6.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["close"] < (dataframe["bb_lowerband"]))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_7_enable.value
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_7.value
                    > dataframe["volume"]
                )
                & (dataframe["ema_26"] > dataframe["ema_12"])
                & (
                    (dataframe["ema_26"] - dataframe["ema_12"])
                    > (dataframe["open"] * self.entry_ema_open_mult_7.value)
                )
                & (
                    (dataframe["ema_26"].shift() - dataframe["ema_12"].shift())
                    > (dataframe["open"] / 100)
                )
                & (dataframe["rsi"] < self.entry_rsi_7.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_8_enable.value
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (dataframe["close"] > dataframe["open"])
                & (dataframe["close"] > dataframe["smma_lips"])
                & (dataframe["smma_lips"] > dataframe["smma_teeth"])
                & (dataframe["smma_teeth"] > dataframe["smma_jaw"])
                & (dataframe["smma_lips"].shift(1) > dataframe["smma_teeth"].shift(1))
                & (dataframe["smma_teeth"].shift(1) > dataframe["smma_jaw"].shift(1))
                & (dataframe["smma_lips"] > dataframe["smma_lips"].shift(1))
                & (dataframe["smma_teeth"] > dataframe["smma_teeth"].shift(1))
                & (dataframe["smma_jaw"] > dataframe["smma_jaw"].shift(1))
                & (dataframe["rsi"] < self.entry_rsi_8.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_9_enable.value
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_30m"])
                & (dataframe["ema_50_30m"] > dataframe["ema_100_30m"])
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_9.value
                    > dataframe["volume"]
                )
                & (dataframe["close"] < dataframe["ema_50"])
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"] * self.entry_bb_offset_9.value
                )
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.entry_condition_10_enable.value
                & (dataframe["sma_200_30m"] > dataframe["sma_200_30m"].shift(24))
                & (dataframe["safe_dips"])
                & (
                    dataframe["volume_mean_4"] * self.entry_volume_10.value
                    > dataframe["volume"]
                )
                & (dataframe["close"] < dataframe["ema_50"])
                & (
                    dataframe["close"]
                    < dataframe["bb_lowerband"] * self.entry_bb_offset_10.value
                )
                & (dataframe["rsi_30m"] < self.entry_rsi_1h_10.value)
                & (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.exit_condition_1_enable.value
                & (dataframe["rsi"] > self.exit_rsi_bb_1.value)
                & (dataframe["close"] > dataframe["bb_upperband"])
                & (dataframe["close"].shift(1) > dataframe["bb_upperband"].shift(1))
                & (dataframe["close"].shift(2) > dataframe["bb_upperband"].shift(2))
                & (dataframe["close"].shift(3) > dataframe["bb_upperband"].shift(3))
                & (dataframe["close"].shift(4) > dataframe["bb_upperband"].shift(4))
                & (dataframe["close"].shift(5) > dataframe["bb_upperband"].shift(5))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_2_enable.value
                & (dataframe["rsi"] > self.exit_rsi_bb_2.value)
                & (dataframe["close"] > dataframe["bb_upperband"])
                & (dataframe["close"].shift(1) > dataframe["bb_upperband"].shift(1))
                & (dataframe["close"].shift(2) > dataframe["bb_upperband"].shift(2))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_3_enable.value
                & (dataframe["rsi"] > self.exit_rsi_main_3.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_4_enable.value
                & (dataframe["rsi"] > self.exit_dual_rsi_rsi_4.value)
                & (dataframe["rsi_30m"] > self.exit_dual_rsi_rsi_1h_4.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_5_enable.value
                & (dataframe["close"] < dataframe["ema_200"])
                & (
                    ((dataframe["ema_200"] - dataframe["close"]) / dataframe["close"])
                    < self.exit_ema_relative_5.value
                )
                & (dataframe["rsi"] > dataframe["rsi_30m"] + self.exit_rsi_diff_5.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_6_enable.value
                & (dataframe["close"] < dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_50"])
                & (dataframe["rsi"] > self.exit_rsi_under_6.value)
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_7_enable.value
                & (dataframe["rsi_30m"] > self.exit_rsi_1h_7.value)
                & (qtpylib.crossed_below(dataframe["ema_12"], dataframe["ema_26"]))
                & (dataframe["volume"] > 0)
            )
        )

        conditions.append(
            (
                self.exit_condition_8_enable.value
                & (
                    dataframe["close"]
                    > dataframe["bb_upperband_30m"] * self.exit_bb_relative_8.value
                )
                & (dataframe["volume"] > 0)
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "exit_long"] = 1

        return dataframe
