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



class EI1_t4c0s_V4(IStrategy):

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

    # ROI table:
    minimal_roi = {
        "0": 0.99,
        
    }

    # Stoploss:
    locked_stoploss = {}
    stoploss = -0.99
    sltime1 = IntParameter(720,1440 , default=720, space='sell', optimize=True)
    sltime2 = IntParameter(1440,2160 , default=720, space='sell', optimize=True)
    sl1x = DecimalParameter(0.01, 0.4, default=0.4, space='sell', optimize=True)
    sl1y = DecimalParameter(0.01, 0.4, default=0.4, space='sell', optimize=True)
    sl1z = DecimalParameter(0.01, 0.4, default=0.4, space='sell', optimize=True)


    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 30, default=21, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(8, 30, default=21, space='sell', optimize=True)
    fib0 = DecimalParameter(0.95, 1.05, default=1.0, decimals=2, space='buy', optimize=True)
    fib1 = DecimalParameter(0.55, 0.65, default=0.62, decimals=2, space='buy', optimize=True)
    fib4 = DecimalParameter(0.70, 0.90, default=0.79, decimals=2, space='buy', optimize=True)
    fib2 = DecimalParameter(0.95, 1.05, default=1.0, decimals=2, space='sell', optimize=True)
    fib3 = DecimalParameter(0.95, 1.05, default=1.0, decimals=2, space='sell', optimize=True)

    # lambo2
    lambo2_rsi_4_limit = IntParameter(30, 60, default=50, space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(30, 55, default=30, space='buy', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    rsi_buy = IntParameter(35, 60, default=50, space='buy', optimize=True)
    move = IntParameter(35, 60, default=48, space='buy', optimize=True)
    mms = IntParameter(6, 20, default=12, space='buy', optimize=True)
    mml = IntParameter(300, 400, default=360, space='buy', optimize=True)

    #cofi
    is_optimize_cofi = True
    buy_ema_cofi = DecimalParameter(0.96, 0.97, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    use_custom_stoploss = True
    process_only_new_candles = True

    # Custom Entry
    last_entry_price = None

    # Unclog
    unclog_days = IntParameter(1, 5, default=4, space='sell', optimize=True)
    unclog = DecimalParameter(0.01, 0.08, default=0.04, decimals=2, space='sell', optimize=True)


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        SLT1 = current_candle['move_mean']
        if trade_duration < self.sltime1.value: 
            SL1 = current_candle['move_mean'] * self.sl1x.value
        if trade_duration > self.sltime1.value and trade_duration < self.sltime2.value: 
            SL1 = current_candle['move_mean'] * self.sl1y.value
        else:    
            SL1 = current_candle['move_mean'] * self.sl1z.value

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

        if exit_reason == 'Down Trend Soon' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} Waiting for Profit")
            self.dp.send_msg(f'{trade.pair} Waiting for Profit')
            return False

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

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < -self.unclog.value and (current_time - trade.open_date_utc).days >= self.unclog_days.value:
            return 'unclog'
    
    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    ## Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    # Optimal timeframe for the strategy
    timeframe = '1h'

    position_adjustment_enable = False
    process_only_new_candles = True
    startup_candle_count = 400


    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # Check how far we are from min and max 
        dataframe['max'] = dataframe['OHLC4'].rolling(self.mms.value).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(self.mms.value).min() / dataframe['OHLC4'] - 1)

        dataframe['max_l'] = dataframe['OHLC4'].rolling(self.mml.value).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(self.mml.value).min() / dataframe['OHLC4'] - 1)

        # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(self.move.value) 
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

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (1-(dataframe['move_mean'] * self.fib0.value))
        dataframe['ma_hi'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (1+(dataframe['move_mean'] * self.fib2.value))
        dataframe['ma_hi_2'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (1+(dataframe['move_mean'] * self.fib3.value))

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


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        lambo2 = (
            (dataframe['close'] < (dataframe['ema_14'] * (1-(dataframe['move_mean'] * self.fib1.value)))) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) &
            (dataframe['atr_pcnt'] > dataframe['min_l']) &
            (dataframe['volume'] > 0) 
        )
        dataframe.loc[lambo2, 'enter_long'] = 1
        dataframe.loc[lambo2, 'enter_tag'] = 'lambo '

        buy1ewo = (
                (dataframe['rsi_fast'] < 35 ) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['close'] < dataframe['enter_mean_x']) &
                (dataframe['close'].shift() < dataframe['enter_mean_x'].shift()) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
        )
        dataframe.loc[buy1ewo, 'enter_long'] = 1
        dataframe.loc[buy1ewo, 'enter_tag'] = 'buy1ewo'

        buy2ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < dataframe['ma_lo']) &
                (dataframe['EWO'] < dataframe['EWO_DN_FIB']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
        )
        dataframe.loc[buy2ewo, 'enter_long'] = 1
        dataframe.loc[buy2ewo, 'enter_tag'] = 'buy2ewo'

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * (1-(dataframe['move_mean'] * self.fib4.value))) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
                (dataframe['atr_pcnt'] > dataframe['min']) &
                (dataframe['volume'] > 0) 
            )
        dataframe.loc[is_cofi, 'enter_long'] = 1
        dataframe.loc[is_cofi, 'enter_tag'] = 'cofi'


        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi_2']) &
                (dataframe['close'] > dataframe['exit_mean_x']) &
                (dataframe['rsi'] > 50 ) &
                (dataframe['volume'] > 0 ) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Close > Offset Hi 2'


        
        condition6 = (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > dataframe['ma_hi']) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
        dataframe.loc[condition6, 'exit_long'] = 1
        dataframe.loc[condition6, 'exit_tag'] = 'Close > Offset Hi 1'

        return dataframe


def pct_change(a, b):
    return (b - a) / a




