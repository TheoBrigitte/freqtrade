
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
import warnings
import pandas as pd
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, informative 
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks, find_peaks_cwt
import warnings
from math import ceil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import time

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
'''



________/\\\\\\\\\__/\\\______________/\\\________________/\\\________/\\\_____/\\\\\\\\\\___/\\\\\_____/\\\__________________________/\\\______________________/\\\\\\\_________________        
 _____/\\\////////__\/\\\__________/\\\\\\\_______________\/\\\_____/\\\//____/\\\///////\\\_\/\\\\\\___\/\\\________________________/\\\\\____________________/\\\/////\\\_______________       
  ___/\\\/___________\/\\\_________\/////\\\_______________\/\\\__/\\\//______\///______/\\\__\/\\\/\\\__\/\\\_____/\\\_____________/\\\/\\\___________________/\\\____\//\\\______________      
   __/\\\_____________\/\\\_____________\/\\\_____/\\\\\\\\_\/\\\\\\//\\\_____________/\\\//___\/\\\//\\\_\/\\\__/\\\\\\\\\\\______/\\\/\/\\\________/\\\\\\\\_\/\\\_____\/\\\__/\\\\\\\\\\_     
    _\/\\\_____________\/\\\\\\\\\\______\/\\\___/\\\//////__\/\\\//_\//\\\___________\////\\\__\/\\\\//\\\\/\\\_\////\\\////_____/\\\/__\/\\\______/\\\//////__\/\\\_____\/\\\_\/\\\//////__    
     _\//\\\____________\/\\\/////\\\_____\/\\\__/\\\_________\/\\\____\//\\\_____________\//\\\_\/\\\_\//\\\/\\\____\/\\\_______/\\\\\\\\\\\\\\\\__/\\\_________\/\\\_____\/\\\_\/\\\\\\\\\\_   
      __\///\\\__________\/\\\___\/\\\_____\/\\\_\//\\\________\/\\\_____\//\\\___/\\\______/\\\__\/\\\__\//\\\\\\____\/\\\_/\\__\///////////\\\//__\//\\\________\//\\\____/\\\__\////////\\\_  
       ____\////\\\\\\\\\_\/\\\___\/\\\_____\/\\\__\///\\\\\\\\_\/\\\______\//\\\_\///\\\\\\\\\/___\/\\\___\//\\\\\____\//\\\\\_____________\/\\\_____\///\\\\\\\\__\///\\\\\\\/____/\\\\\\\\\\_ 
        _______\/////////__\///____\///______\///_____\////////__\///________\///____\/////////_____\///_____\/////______\/////______________\///________\////////_____\///////_____\//////////__





'''




pd.set_option('display.float_format', lambda x: '%.7f' % x)
logger = logging.getLogger(__name__)

class ARIMA_15(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.195,
        "360": 0.10600000000000001,
        "910": 0.04,
        "1440": 0.005
    }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    use_custom_stoploss = True

    # Initialize dicts for arima storage
    last_run_time = {} 
    arima_model = {} 
    last_run_time_1h = {} 
    arima_model_1h = {} 
    last_run_time_4h = {} 
    arima_model_4h = {} 

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
    timeframe = '15m'
    startup_candle_count = 400
    process_only_new_candles = True

    # Custom Entry
    last_entry_price = None

    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True)
    up = DecimalParameter(low=1.020, high=1.025, default=1.02, decimals=3 ,space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3 ,space='buy', optimize=True, load=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4 ,space='buy', optimize=True, load=True)
    atr_length = IntParameter(5, 30, default=5, space='buy', optimize=True, load=True)
    window = IntParameter(10, 30, default=16, space='buy', optimize=True, load=True)
    window_1h = IntParameter(10, 30, default=8, space='buy', optimize=True, load=True)
    window_4h = IntParameter(2, 30, default=2, space='buy', optimize=True, load=True)
    x = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2 ,space='buy', optimize=True, load=True)
    x_1h = DecimalParameter(low=1.2, high=1.75, default=1.5, decimals=2 ,space='buy', optimize=True, load=True)
    x_4h = DecimalParameter(low=1.2, high=1.75, default=1.3, decimals=2 ,space='buy', optimize=True, load=True)


    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.10, high=0.15, default=0.15, decimals=2,  space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3,  space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.10, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.06, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)


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


    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle['max_l'] > 0.0035:


            if (current_profit > self.tsl_target3.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl3 {self.tsl_target3.value}/{self.ts3.value} activated')
                return self.ts3.value
            if (current_profit > self.tsl_target2.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl2 {self.tsl_target2.value}/{self.ts2.value} activated')
                return self.ts2.value 
            if (current_profit > self.tsl_target1.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl1 {self.tsl_target1.value}/{self.ts1.value} activated')
                return self.ts1.value
            if (current_profit > self.tsl_target0.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                logger.info(f'*** {pair} *** Profit {current_profit} - lvl0 {self.tsl_target0.value}/{self.ts0.value} activated')
                return self.ts0.value 

        else:
            if (current_profit > self.tsl_target0.value):
                self.dp.send_msg(f'*** {pair} *** Profit {current_profit} SWINGING FOR THE MOON!!!')
                return 0.99

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

        if exit_reason == 'roi' and (last_candle['min_l'] > (last_candle['max_l'] * 3)):
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


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['decision'] = 0
        pair = metadata['pair']
        current_time = time.time()

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        size = len(dataframe) - 10
        train, test = model_selection.train_test_split(dataframe['OHLC4'], train_size=size)

        # Initialize values for the current pair if not already done
        if pair not in self.last_run_time:
            self.last_run_time[pair] = current_time

            logger.info(f"Initial ARIMA {self.timeframe} Model Training for {pair}")
            # Fit ARIMA model
            start_time = time.time()
            self.arima_model[pair] = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                 stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                 error_action='ignore')
            fitting_time = time.time() - start_time
            logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")


        # Check if it's time to retrain for the current pair
        if current_time - self.last_run_time[pair] >= 3600:  # Check if an hour has passed
            logger.info(f"Auto Fitting ARIMA {self.timeframe} Model for {pair}")
            self.last_run_time[pair] = current_time

            # Fit ARIMA model
            start_time = time.time()
            self.arima_model[pair] = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                                                 max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                                                 stepwise=True, suppress_warnings=True, D=10, max_D=20,
                                                 error_action='ignore')
            fitting_time = time.time() - start_time
            logger.info(f"{pair} - ARIMA {self.timeframe} Model fitted in {fitting_time:.2f} seconds")

        # Use the previously fitted ARIMA model for forecasting
        if self.arima_model[pair] is not None:
            start_time = time.time()
            future_forecast, conf_int = self.arima_model[pair].predict(n_periods=test.shape[0], return_conf_int=True)
            inference_time = time.time() - start_time

        timeleft = current_time - self.last_run_time[pair]
        if timeleft <= 3600 and timeleft != 0:
            logger.info(f"{pair} - ARIMA {self.timeframe} Model re-optimized in {timeleft:.2f} seconds")

        # Extract upper and lower confidence intervals
        lower_confidence, upper_confidence = conf_int[:, 0], conf_int[:, 1]
        logger.info(f"{pair} - Inference time: {inference_time:.2f} seconds | " \
             f"Current Price: {dataframe['OHLC4'].iloc[-1]:.7f} | {self.timeframe} Future Forecast: {future_forecast.iloc[-1]:.7f}")

        dataframe['rmse'] = 0
        dataframe['accuracy_perc'] = 0
        dataframe['reward'] = 0
        dataframe['rmse'] = np.sqrt(mean_squared_error(test, future_forecast))
        dataframe['accuracy_perc'] = 100 * (1 - (dataframe['rmse'].iloc[-1] / dataframe['OHLC4'].iloc[-1]))
        dataframe['reward'] = ((future_forecast.iloc[-1] / dataframe['OHLC4'].iloc[-1]) - 1) * 100

        rmse = dataframe['rmse'].iloc[-1] 
        accuracy_perc = dataframe['accuracy_perc'].iloc[-1] 
        reward = dataframe['reward'] .iloc[-1] 
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value) # 5.25 hrs

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x.value
        move = '{:.2f}'.format(dataframe['move'].iloc[-1] * 100)
        move_mean = '{:.2f}'.format(dataframe['move_mean'].iloc[-1] * 100)

        if future_forecast.iloc[-1] > dataframe['OHLC4'].iloc[-1]:

        
            direction = 'Up'
            dataframe['decision'] = 1
        else:
            direction = 'Down'
            dataframe['decision'] = -1
        


        logger.info(f"{pair} - Test RMSE: {rmse:.3f} | Accuracy: {accuracy_perc:.2f}% | Potential Profit: {move}% | Avg. Profit: {move_mean}% | {self.timeframe} Trend: {direction}")

        dataframe['arima_predictions'] = pd.Series(future_forecast)
        dataframe['lower_confidence'] = pd.Series(lower_confidence)
        dataframe['upper_confidence'] = pd.Series(upper_confidence)
        
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=self.atr_length.value) / dataframe['OHLC4'])
        dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value

        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)

        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition1 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min_l'] < dataframe['max_l']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['OHLC4'] < dataframe['sma_dn']) &
            (dataframe['sma_dn'].shift() > dataframe['sma_dn']) &

            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition1, 'enter_long'] = 1
        dataframe.loc[condition1, 'enter_tag'] = 'Up Trend Soon below sma_dn'

        condition2 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean_x']) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min_l'] < dataframe['max_l']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['OHLC4'] < dataframe['sma_dn']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition2, 'enter_long'] = 1
        dataframe.loc[condition2, 'enter_tag'] = 'Move Mean Fib below sma_dn'

        condition3 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min_l'] < dataframe['max_l']) &
            (dataframe['OHLC4'] < dataframe['sma']) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition3, 'enter_long'] = 1
        dataframe.loc[condition3, 'enter_tag'] = 'Up Trend Soon below sma'

        condition4 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean_x']) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['min_l'] < dataframe['max_l']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['OHLC4'] < dataframe['sma']) &
            (dataframe['max_l'] < dataframe['atr_pcnt']) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition4, 'enter_long'] = 1
        dataframe.loc[condition4, 'enter_tag'] = 'Move Mean Fib below sma'

        condition5150 = (
            (dataframe['decision'] == 1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['min'] < dataframe['max']) &
            (dataframe['OHLC4'] > dataframe['sma']) &
            (dataframe['sma_up'].shift() < dataframe['sma']) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[condition5150, 'enter_long'] = 1
        dataframe.loc[condition5150, 'enter_tag'] = 'Hope this works...'

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        condition5 = (
            (dataframe['decision'] == -1) &
            (dataframe['move'] >= dataframe['move_mean']) &
            (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) &
            (dataframe['min'] > dataframe['max']) &
            (dataframe['min_l'] > dataframe['max_l']) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Down Trend Soon'

        condition6 = (
            (dataframe['decision'] == -1) &
            (dataframe['move'] >= dataframe['move_mean_x']) &
            (dataframe['move'].shift(3) >= dataframe['move_mean_x'].shift(3)) &
            (dataframe['min'] > dataframe['max']) &
            (dataframe['min_l'] > dataframe['max_l']) &
            (dataframe['volume'] > 0)
            )

        dataframe.loc[condition6, 'exit_long'] = 1
        dataframe.loc[condition6, 'exit_tag'] = 'Move Mean Fib'

        return dataframe

