# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import logging
import math
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import time
from typing import Dict, List, Optional

# ---
import sys
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
import technical.indicators as ftt


######################################## Warning ########################################
# You won't get a lot of benefits by simply changing to this strategy                   #
# with the HyperOpt values changed.                                                     #
#                                                                                       #
# You should test it closely, trying backtesting and dry running, and we recommend      #
# customizing the terms of sale and purchase as well.                                   #
#                                                                                       #
# You should always be careful in real trading!                                         #
#########################################################################################


def EWO(dataframe, ema_length=5, ema2_length=35):
    # df = dataframe.copy()
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / dataframe['close'] * 100
    return emadif


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    return (1 - t) * a + t * b


logger = logging.getLogger(__name__)


class SMAOffset_Hippocritical_dca(IStrategy):
    # Original: SMAOffsetProtectOptV1 by kkeue, shared on the freqtrade discord at 2021-06-19
    # Added dca of Stash86, modified and optimized it.
    # Added jorikito#2815 's partial fill compensation

    # Use this option with caution!
    # enables a full 1st slot and base+safety-order for the 2nd slot before you run out of money in your wallet.
    # (combo from the .ods-file: 4+3, see rows "Overbuy calculation")

    overbuy_factor = 1.295

    position_adjustment_enable = True
    initial_safety_order_trigger = -0.02
    max_so_multiplier_orig = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 1.8

    # just for initialization, now we calculate it...
    max_so_multiplier = max_so_multiplier_orig
    # We will store the size of stake of each trade's first order here
    cust_proposed_initial_stakes = {}
    # Amount the strategy should compensate previously partially filled orders for successive safety orders (0.0 - 1.0)
    partial_fill_compensation_scale = 1

    if (max_so_multiplier_orig > 0):
        if (safety_order_volume_scale > 1):
            # print(safety_order_volume_scale * (math.pow(safety_order_volume_scale,(max_so_multiplier - 1)) - 1))

            firstLine = (safety_order_volume_scale *
                         (math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)) - 1))
            divisor = (safety_order_volume_scale - 1)
            max_so_multiplier = (2 + firstLine / divisor)
            # max_so_multiplier = (2 +
            #                     (safety_order_volume_scale *
            #                      (math.pow(safety_order_volume_scale, (max_so_multiplier - 1)) - 1) /
            #                      (safety_order_volume_scale - 1)))
        elif (safety_order_volume_scale < 1):
            firstLine = safety_order_volume_scale * \
                        (1 - math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)))
            divisor = 1 - safety_order_volume_scale
            max_so_multiplier = (2 + firstLine / divisor)
            # max_so_multiplier = (2 + (safety_order_volume_scale * (
            #        1 - math.pow(safety_order_volume_scale, (max_so_multiplier - 1))) / (
            #                                  1 - safety_order_volume_scale)))

    # Since stoploss can only go up and can't go down, if you set your stoploss here, your lowest stoploss will always be tied to the first buy rate
    # So disable the hard stoploss here, and use custom_sell or custom_stoploss to handle the stoploss trigger
    stoploss = -1

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            return tag

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        if current_profit <= -0.35:
            return f'stop_loss ({buy_tag})'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        # remove pair from custom initial stake dict only if full exit
        if trade.amount == amount:
            del self.cust_proposed_initial_stakes[pair]
        return True

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
        self.cust_proposed_initial_stakes[
            pair] = custom_stake  # Setting of first stake size just before each first order of a trade
        return custom_stake # set to static 10 to simulate partial fills of 10$, etc

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:
        if current_profit > self.initial_safety_order_trigger:
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if 1 <= count_of_buys <= self.max_so_multiplier_orig:
            # if (1 <= count_of_buys) and (open_trade_value < self.stake_amount * self.overbuy_factor):
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                        abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (
                                self.safety_order_step_scale - 1))
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                        abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / (
                                1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    # This returns first order actual stake size
                    actual_initial_stake = filled_buys[0].cost

                    # Fallback for when the initial stake was not set for whatever reason
                    stake_amount = actual_initial_stake

                    already_bought = sum(filled_buy.cost for filled_buy in filled_buys)

                    if self.cust_proposed_initial_stakes[trade.pair] > 0:
                        # This calculates the amount of stake that will get used for the current safety order,
                        # including compensation for any partial buys
                        proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                        current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale,
                                                                         (count_of_buys - 1))
                        current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale,
                                                                                      (count_of_buys - 1))
                        current_stake_preposition_compensation = current_stake_preposition + abs(
                            current_stake_preposition - current_actual_stake)
                        total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation,
                                              self.partial_fill_compensation_scale)
                        # Set the calculated stake amount
                        stake_amount = total_so_stake
                    else:
                        # Fallback stake amount calculation
                        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))

                    amount = stake_amount / current_rate
                    logger.info(
                        f"Initiating safety order buy #{count_of_buys} "
                        f"for {trade.pair} with stake amount of {stake_amount}. "
                        f"which equals {amount}. "
                        f"Previously bought: {already_bought}. "
                        f"Now overall:{already_bought + stake_amount}. ")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    # print(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None

    # Modified Buy / Sell params - 20210619
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 16,
        "ewo_high": 5.672,
        "ewo_low": -19.931,
        "low_offset": 0.973,
        "rsi_buy": 59,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.010,
    }
    INTERFACE_VERSION = 2

    # Modified ROI - 20210620
    # ROI table:
    minimal_roi = {
        "0": 0.028,
        "10": 0.018,
        "30": 0.010,
        "40": 0.005
    }

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count: int = 576

    plot_config = \
        {
            "main_plot":
                {},
            "subplots":
                {
                    "sub":
                        {
                            "rsi":
                                {
                                    "color": "#80dea8",
                                    "type": "line"
                                }
                        },
                    "sub2":
                        {
                            "ma_buy_16":
                                {
                                    "color": "#db1ea2",
                                    "type": "line"
                                },
                            "ma_sell_20":
                                {
                                    "color": "#645825",
                                    "type": "line"
                                },
                            "EWO":
                                {
                                    "color": "#1e5964",
                                    "type": "line"
                                },
                            "missing_data":
                                {
                                    "color": "#26b08d",
                                    "type": "line"
                                }
                        }
                }
        }

    use_custom_stoploss = False

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        # moved into entry / exit functions to not have redundant calculations being done.
        # backtesting will be slower but who cares about a few seconds during backtests right?
        # All that counts is dry / live performance, not backtesting speed imo ...
        # If you want to put in the work with conditions be my guest ;)
        """
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        """
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Check for 0 volume candles in the last day
        dataframe['missing_data'] = \
            (dataframe['volume'] <= 0).rolling(
                window=self.startup_candle_count,
                min_periods=self.startup_candle_count).sum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Speed optimization for dry / live runs, not looping through for ... values with it, nothing else.
        dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] = \
            ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)

        conditions = []
        conditions.append(
            (dataframe['close'] < (
                    dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['missing_data'] < 1)
        )
        conditions.append(
            (dataframe['close'] < (
                    dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['missing_data'] < 1)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Speed optimization for dry / live runs, not looping through for ... values with it, nothing else.
        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] = \
            ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)

        conditions = []

        conditions.append(
            (
                    (dataframe['close'] > (
                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1

        return dataframe