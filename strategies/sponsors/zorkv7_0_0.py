from datetime import datetime, timedelta
from typing import Optional, Union
import random

import talib.abstract as ta
import pandas as pd
from pandas import DataFrame
import numpy as np

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter, merge_informative_pair
from freqtrade.exchange import timeframe_to_prev_date
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import zema, vfi
from sklearn import preprocessing

import freqtrade.vendor.qtpylib.indicators as qtpylib

# Hyperopt instructions
# 0) set ROI to "0": 1, set stoploss to -0.3, and comment out trailing stoploss
# 1) hyperopt on buy and sell spaces only and update params
# 2) hyperopt on ROI only and update table
# 3) hyperopt on stoploss and update
# 4) hyperopt on trailing only if required
#
# author @jimmynixx and his special sauce - inspired by @rextea, rk2 version by @rk, roi and custom_sell from @your_secret_admirer, zema7 exit check from @zx, munged into being by @froggleston with Solipsis inspiration and memey goodness from @werkkrew

class zorkv7_0_0(IStrategy):
    # ROI table:
    minimal_roi = {
        "0":1
        # "0": 0.078,
        # "40": 0.062,
        # "99": 0.039,
        # "218": 0
    }

    stoploss = -0.1

    # Buy hyperspace params:
    buy_params = {
        'low_offset': 0.964,
        'wma_low_offset': 0.979,
        'ma_len_buy': 51,
        'buy_ma_type': 'zema',
    }

    # Sell hyperspace params:
    # sell_params = {
    #     "max_roi_time": 95,
    #     "max_stoploss_time": 204,
    #     "roi_factor_bear": 1.145,
    #     "roi_factor_bull": 6.21,
    #     "sl_factor_bear": 1.054,
    #     "sl_factor_bull": 9.22,
    #     "high_offset": 1.004,  # value loaded from strategy
    #     "ma_len_sell": 72,  # value loaded from strategy
    #     "msq_normabs_sell": 1.95,  # value loaded from strategy
    #     "sell_ma_type": "zema",  # value loaded from strategy
    #     "trail_amount": 0.006,  # value loaded from strategy
    #     "wma_high_offset": 1.04,  # value loaded from strategy
    # }

    # Sell hyperspace params:
    sell_params = {
        "max_roi_time": 36,
        "max_stoploss_time": 65,
        "roi_factor_bear": 4.045,
        "roi_factor_bull": 4.75,
        "sl_factor_bear": 2.932,
        "sl_factor_bull": 8.648,
        "high_offset": 1.004,  # value loaded from strategy
        "ma_len_sell": 72,  # value loaded from strategy
        "msq_normabs_sell": 1.95,  # value loaded from strategy
        "sell_ma_type": "zema",  # value loaded from strategy
        "trail_amount": 0.006,  # value loaded from strategy
        "wma_high_offset": 1.04,  # value loaded from strategy
    }




    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.02 # 0.02
    trailing_stop_positive_offset = 0.045 # 0.045
    trailing_only_offset_is_reached = False

    # Custom stoploss
    use_custom_stoploss = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    timeframe = '5m'
    informative_timeframe = '5m'
    pair_informative_timeframe = '1h'
    informative_coin = "ETH"

    high_offset = DecimalParameter(1, 1.20, default=1.004, space='sell', optimize=False)
    wma_high_offset = DecimalParameter(1, 1.20, default=1.01, space='sell', optimize=False)
    low_offset = DecimalParameter(0.80, 0.99, default=0.974, space='buy', optimize=False)
    wma_low_offset = DecimalParameter(0.80, 0.99, default=0.944, space='buy', optimize=False)
    ma_len_buy = IntParameter(30, 90, default=72, space='buy', optimize=False)
    ma_len_sell = IntParameter(30, 90, default=51, space='sell', optimize=False)
    buy_ma_type = CategoricalParameter(['zema', 'wma'], space='buy', default='zema', optimize=False)
    sell_ma_type = CategoricalParameter(['zema', 'wma'], space='sell', default='zema', optimize=False)
    msq_normabs_sell = DecimalParameter(1.0, 4.0, default=2, space='sell', optimize=False)

    roi_factor_bear = DecimalParameter(0.5, 5, default=2, space='sell', optimize=True, load=True)
    roi_factor_bull = DecimalParameter(0.5, 5, default=2, space='sell', optimize=True, load=True)
    sl_factor_bear = DecimalParameter(1, 10, default=3, space='sell', optimize=True, load=True)
    sl_factor_bull = DecimalParameter(1, 10, default=3, space='sell', optimize=True, load=True)
    max_stoploss_time = IntParameter(30, 240, default=100, space='sell', optimize=True, load=True)
    max_roi_time = IntParameter(30, 240, default=100, space='sell', optimize=True, load=True)

    trail_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=False)

    market_df = None

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'green'},
            'ma_sell': {'color': 'red'},
        },
        'subplots': {
        }
    }

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.pair_informative_timeframe) for pair in pairs]
        return informative_pairs

    def inf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.pair_informative_timeframe)
        informative['ema_50'] = ta.EMA(informative, timeperiod=50)
        informative['ema_200'] = ta.EMA(informative, timeperiod=200)

        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pinformative = self.inf_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, pinformative, self.timeframe, self.pair_informative_timeframe, ffill=True)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['zema7'] = zema(dataframe, period=7)
        dataframe['zema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['wma100'] = ta.WMA(dataframe, timeperiod=300)

        informative = self.dp.get_pair_dataframe(f'{self.informative_coin}/USDT', self.informative_timeframe)
        informative['sma50'] = ta.SMA(informative, timeperiod=50)
        informative['sma100'] = ta.SMA(informative, timeperiod=100)
        informative['sma200'] = ta.SMA(informative, timeperiod=200)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        # Bollinger Bands because obviously
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=1)
        informative['bb_lowerband'] = bollinger['lower']
        informative['bb_middleband'] = bollinger['mid']
        informative['bb_upperband'] = bollinger['upper']

        # informative SAR        if roi_decay<0.005: roi_estimate = 0.005
        informative[f'{self.informative_coin}_sar'] = ta.SAR(informative)

        informative[f'{self.informative_coin}_dmi_plus'] = ta.PLUS_DI(informative, timeperiod=14)
        informative[f'{self.informative_coin}_dmi_minus'] = ta.MINUS_DI(informative, timeperiod=14)
        informative[f'{self.informative_coin}_adx'] = ta.ADX(informative, timeperiod=9)

        informative[f'{self.informative_coin}_roc'] = ta.ROC(informative, timeperiod=6)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

        for length in self.ma_len_buy.range:
            dataframe[f'zema_{length}'] = zema(dataframe, period=length)
            dataframe[f'wma_{length}'] = ta.WMA(dataframe, timeperiod=length)
        for length in self.ma_len_sell.range:
            if f'zema_{length}' not in dataframe:
                dataframe[f'zema_{length}'] = zema(dataframe, period=length)
                dataframe[f'wma_{length}'] = ta.WMA(dataframe, timeperiod=length)

        dataframe['zema_buy'] = dataframe[f'zema_{self.ma_len_buy.value}'] * self.low_offset.value
        dataframe['zema_sell'] = dataframe[f'zema_{self.ma_len_sell.value}'] * self.high_offset.value
        dataframe['wma_buy'] = dataframe[f'wma_{self.ma_len_buy.value}'] * self.wma_low_offset.value
        dataframe['wma_sell'] = dataframe[f'wma_{self.ma_len_sell.value}'] * self.wma_high_offset.value

        msq_normabs = self.MadSqueeze(dataframe, period=14, ref=13, sqzlen=5)
        dataframe['msq_normabs'] = msq_normabs

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['mfi'] = ta.MFI(dataframe)

        #dataframe_macro = resample_to_interval(dataframe, self.get_ticker_indicator())
        dataframe['high-low'] = dataframe['high'] - dataframe['low']
        dataframe['open-close'] = dataframe['open'] - dataframe['close']
        dataframe['avg_candle_size'] = dataframe['high-low'].rolling(20).mean()

        dataframe['pct_change'] = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close']
        #dataframe = resampled_merge(dataframe, dataframe_macro)

        dataframe = self.custom_roi_stoploss_bmsb_ATR(dataframe,12)

        dataframe = self.get_market_trend(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ma_buy = f'{self.buy_ma_type.value}_{self.ma_len_buy.value}'

        outside_range = 4
        dataframe.loc[
            (
                ## JIMMY'S SPECIAL SAUCE DO NOT TOUCH
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h'])
                &
                (dataframe[f'{self.informative_coin}_dmi_minus_5m'] > dataframe[f'{self.informative_coin}_dmi_plus_5m'])
                &
                (dataframe[f'{self.informative_coin}_dmi_minus_5m'] > 25)
                &
                (dataframe[f'{self.informative_coin}_dmi_plus_5m'] < 10)
                &
                (
                    (
                        ((dataframe['open'] < (dataframe[ma_buy] * self.low_offset.value)).rolling(window=outside_range).sum() == outside_range)
                        &
                        ((dataframe['close'] < (dataframe[ma_buy] * self.low_offset.value)).rolling(window=outside_range).sum() == outside_range)
                        &
                        (dataframe['close'] < dataframe[f'{self.buy_ma_type.value}100'])
                    )
                    |
                    (
                        (dataframe['close'] < (dataframe[ma_buy] * self.low_offset.value))
                        &
                        (dataframe['close'] > dataframe[f'{self.buy_ma_type.value}100'])
                    )
                )
                &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        ## comment out to be less strict
        dataframe.loc[
            (
                ## don't buy shitecoins - thanks @rextea!
                (dataframe['up-ratio'] <= 0.15)
                |
                # low activity sideways periods - reduce to allow more buys
                (dataframe['msq_normabs'] < 1.2)
                |
                (
                    # wacky shit happening
                    (dataframe['zema_sell'] > dataframe['wma_sell'])
                    |
                    (dataframe['zema_buy'] > dataframe['wma_buy'])
                )
                &
                (
                    # informative guards (BTC/ETH)
                    (dataframe[f'{self.informative_coin}_roc_5m'] < -5)
                    &
                    (dataframe[f'{self.informative_coin}_roc_5m'] < dataframe[f'{self.informative_coin}_roc_5m'].shift())
                )
            ),'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ma_sell = f'{self.sell_ma_type.value}_{self.ma_len_sell.value}'

        #stock zemaoff_rk sell
        dataframe.loc[
            (
                (dataframe['close'] > (dataframe[ma_sell] * self.high_offset.value))
                &
                (dataframe['close'] < dataframe[f'{self.sell_ma_type.value}100'])
                &
                (dataframe['volume'] > 0)

            ),
            'sell'] = 1

        ## comment out to be less strict
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['wma100'])
                |
                (
                    (dataframe['close'] < dataframe['zema_sell'])
                    &
                    (dataframe['close'] < dataframe['sar'])
                )
            ),
            'sell'] = 0

        return dataframe

    ## drafty's custom_sell from golden_corralation
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        current_profit = trade.calc_profit_ratio(current_candle['close'])

        trade_date = timeframe_to_prev_date("5m",trade.open_date_utc)
        trade_candle = dataframe.loc[(dataframe['date'] == trade_date)]

        if trade_candle.empty: return None
        trade_candle = trade_candle.squeeze()

        open_price = (1 - current_profit) * current_rate
        roi_price =  trade_candle['roi_price']
        roi_estimate = ( roi_price / open_price ) - 1
        if roi_estimate > 0.08: roi_estimate=0.08
        custom_roi = roi_estimate

        stoploss_price =  trade_candle['custom_stop_loss']
        if stoploss_price<0: stoploss_price = -stoploss_price
        stoploss_estimate = 1 - (stoploss_price / open_price)
        if stoploss_estimate>-self.stoploss: stoploss_estimate = -self.stoploss
        stoploss_decay = stoploss_estimate * ( 1 - ((current_time - trade.open_date_utc).seconds) / (self.max_stoploss_time.value*60)) # linear decay
        if (current_profit < -stoploss_decay):
            if trade_candle['resample_{}_bmsb'.format(self.get_ticker_indicator()*12)] == 0:
                return 'custom_bear_stop_loss'
            else: return 'custom_bull_stop_loss'

        roi_decay = roi_estimate * ( 1 - ((current_time - trade.open_date_utc).seconds) / (self.max_roi_time.value*60)) # linear decay
        #roi_decay = roi_estimate * np.exp(-(current_time - trade.open_date_utc).seconds) / (120*60) # exponential decay
        if roi_decay < 0.005: roi_estimate = 0.005
        else: roi_estimate = roi_decay

        if current_profit > roi_estimate:
            if trade_candle['resample_{}_bmsb'.format(self.get_ticker_indicator()*12)] == 0:
                return 'roi_custom_bear'
            else: return 'roi_custom_bull'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # if rate < trade.stop_loss: return True
        if (last_candle is not None):
            current_profit = trade.calc_profit_ratio(rate)
            max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
            pullback_value = max(0, (max_profit - self.trail_amount.value))

            if (sell_reason in ('force_sell', 'sell_signal')) or (sell_reason.endswith('stop_loss')):
                return True

            # Ignore ROI if seems to go up:
            if sell_reason.startswith('roi'):
                if last_candle['buy']: return False

                if (current_profit >= pullback_value):
                    if (last_candle['rsi'] > 40):
                        if (last_candle['close'] > last_candle['wma100']) & (last_candle['close'] > last_candle['zema7']):
                            return False

                        if (last_candle['mfi'] < 70):
                            return False
                return True
            return False
        return True

## comment in for @rextea's random coin purse sampler
#    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
#                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
#        rand = random.randint(1, 10)
#        if rand == 1:
#            return False
#
#        return True

    ##

    def custom_roi_stoploss_bmsb_ATR(self, dataframe, window):

        dataframe_macro = resample_to_interval(dataframe, self.get_ticker_indicator()*window) # 1 hr
        dataframe_macro['ATR'] = ta.ATR(dataframe_macro, timeperiod=14)
        dataframe_macro['20sma'] = ta.SMA(dataframe_macro,timeperiod=20)
        dataframe_macro['21ema'] = ta.EMA(dataframe_macro,timeperiod=21)
        dataframe_macro['bmsb'] = np.where((dataframe_macro['20sma'] > dataframe_macro['21ema']) & (dataframe_macro['close'] > dataframe_macro['20sma']), 1, 0)
        dataframe = resampled_merge(dataframe, dataframe_macro)

        dataframe['custom_stop_loss'] = np.where(dataframe['resample_{}_bmsb'.format(self.get_ticker_indicator()*window)] == 0, dataframe['low'] - dataframe['resample_{}_ATR'.format(self.get_ticker_indicator()*window)] * self.sl_factor_bear.value, dataframe['low'] - dataframe['resample_{}_ATR'.format(self.get_ticker_indicator()*window)] * self.sl_factor_bull.value)
        dataframe['roi_price'] = np.where(dataframe['resample_{}_bmsb'.format(self.get_ticker_indicator()*window)] == 0, dataframe['close'] + dataframe['resample_{}_ATR'.format(self.get_ticker_indicator()*window)] * self.roi_factor_bear.value, dataframe['close'] + dataframe['resample_{}_ATR'.format(self.get_ticker_indicator()*window)] * self.roi_factor_bull.value)

        return dataframe


    def get_market_trend(self, dataframe):
        pairs = self.dp.current_whitelist()
        top20 = pairs[:20]

        if self.market_df is None:
            self.market_df = dataframe.copy()[['date']]
            self.market_df['up'] = 0
            for pair in top20:
                df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
                df['up'] = np.where(df['close'] > df['close'].shift(12), 1, 0)
                self.market_df['up'] = self.market_df['up'].add(df['up'])

        self.market_df['up-ratio'] = self.market_df['up'] / len(top20)
        dataframe = pd.merge(dataframe, self.market_df[['date', 'up', 'up-ratio']], on='date', how='left')
        return dataframe

    def SSLChannels_ATR(self, dataframe, length=7):
        """
        SSL Channels with ATR: https://www.tradingview.com/script/SKHqWzql-SSL-ATR-channel/
        Credit to @JimmyNixx for python
        """
        df = dataframe.copy()

        df['ATR'] = ta.ATR(df, timeperiod=14)
        df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
        df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
        df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
        df['hlv'] = df['hlv'].ffill()
        df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
        df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])

        return df['sslDown'], df['sslUp']

    def SROC(self, dataframe, roclen=21, emalen=13, smooth=21):
        df = dataframe.copy()

        roc = ta.ROC(df, timeperiod=roclen)
        ema = ta.EMA(df, timeperiod=emalen)
        sroc = ta.ROC(ema, timeperiod=smooth)

        return sroc

    ## based on https://www.tradingview.com/script/9bUUSzM3-Madrid-Trend-Squeeze/
    def MadSqueeze(self, dataframe, period=34, ref=13, sqzlen=5):
        df = dataframe.copy()

        # min period force
        if period < 14:
            period = 14

        ma = ta.EMA(df['close'], period)

        closema = df['close'] - ma
        df['msq_closema'] = closema

        refma = ta.EMA(df['close'], ref) - ma
        df['msq_refma'] = refma

        sqzma = ta.EMA(df['close'], sqzlen) - ma
        df['msq_sqzma'] = sqzma

        ## Apply a non-parametric transformation to map the Madrid Trend Squeeze data to a Gaussian.
        ## We do this to even out the peaks across the dataframe, and end up with a normally distributed measure of the variance
        ## between ma, the reference EMA and the squeezed EMA
        ## The bigger the number, the bigger the peak detected. Buys and sells tend to happen at the peaks.

        # max samples 100000
        n_quants = 100000 if df.shape[0] >= 100000 else df.shape[0]-1

        quantt = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0, n_quantiles=n_quants)
        df['msq_abs'] = (df['msq_closema'].fillna(0).abs() + df['msq_refma'].fillna(0).abs() + df['msq_sqzma'].fillna(0).abs())
        df['msq_normabs'] = quantt.fit_transform(df[['msq_abs']])

        return df['msq_normabs']
