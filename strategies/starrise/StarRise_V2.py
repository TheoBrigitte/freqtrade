# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import talib.abstract as ta
import numpy as np
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
import math
import logging
import datetime
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
# --------------------------------

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
# Williams %R

def williams_r(dataframe: DataFrame, period: int=14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """
    highest_high = dataframe['high'].rolling(center=False, window=period).max()
    lowest_low = dataframe['low'].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe['close']) / (highest_high - lowest_low), name=f'{period} Williams %R')
    return WR * -100

class StarRise_strat(IStrategy):
    INTERFACE_VERSION = 3
    '\n\n    Designed to use with StarRise DCA settings\n\n    TTP: 1.1%(0.2%), BO: 38.0 USDT, SO: 38.0 USDT, OS: 1.2, SS: 1.13, MAD: 2, SOS: 1.6, MSTC: 11\n\n\n    2021/12 Crash\n        ========================================================== BUY TAG STATS ===========================================================\n    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |\n    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|\n    | TOTAL |    412 |           1.14 |         469.43 |          1157.492 |           0.45 |        5:04:00 |   412     0     0   100 |\n\n    2021/05 Crash\n        ========================================================== BUY TAG STATS ===========================================================\n    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |\n    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|\n    | TOTAL |    197 |           1.25 |         245.79 |           631.840 |           0.25 |        4:22:00 |   197     0     0   100 |\n\n    2021/09 - 2021/11 Bull\n        ========================================================== BUY TAG STATS ===========================================================\n    |   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |\n    |-------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|\n    | TOTAL |    327 |           1.30 |         424.98 |           961.187 |           0.37 |        3:26:00 |   326     0     1  99.7 |\n\n    '
    # Minimal ROI designed for the strategy.
    minimal_roi = {'0': 0.005}
    # Sell hyperspace params:
    # 1.1% TTP
    sell_params = {'pHSL': -0.18, 'pPF_1': 0.019, 'pPF_2': 0.054, 'pSL_1': 0.019, 'pSL_2': 0.053}
    # Max Deviation -0.349
    stoploss = -0.20
    # Custom stoploss
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 168
    # Optimal timeframe for the strategy
    timeframe = '5m'
    # hard stoploss profit
    pHSL = DecimalParameter(-0.5, -0.04, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.02, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.02, default=0.011, decimals=3, space='sell', load=True)
    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.04, 0.1, default=0.08, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.02, 0.07, default=0.04, decimals=3, space='sell', load=True)
    use_exit_signal = True
    exit_profit_only = True
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
        if current_profit > 0.2:
            if last_candle['rsi'] < 80:
                return 'rsi_below_80'

        

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 2:
            return 'unclog'
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
        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + (current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1)
        else:
            sl_profit = HSL
        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99
        return stoploss_from_open(sl_profit, current_profit)

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # CTI
        dataframe['cti_40'] = pta.cti(dataframe['close'], length=40)
        # %R
        dataframe['r_96'] = williams_r(dataframe, period=96)
        dataframe['r_480'] = williams_r(dataframe, period=480)
        # 1h mama > fama for general trend check
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)
        dataframe['mama_diff'] = (dataframe['mama'] - dataframe['fama']) / dataframe['hl2']
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)
        # Bollinger bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=17, stds=1)
        dataframe['bb_lowerband'] = bollinger1['lower']
        dataframe['bb_middleband'] = bollinger1['mid']
        dataframe['bb_upperband'] = bollinger1['upper']
        # Close delta
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        # Dip Protection
        dataframe['tpct_change_0'] = top_percent_change(dataframe, 0)
        dataframe['tpct_change_1'] = top_percent_change(dataframe, 1)
        dataframe['tpct_change_2'] = top_percent_change(dataframe, 2)
        dataframe['tpct_change_4'] = top_percent_change(dataframe, 4)
        dataframe['tpct_change_5'] = top_percent_change(dataframe, 5)
        dataframe['tpct_change_9'] = top_percent_change(dataframe, 9)
        # SMA
        dataframe['sma_50'] = ta.SMA(dataframe['close'], timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe['close'], timeperiod=200)
        # CTI
        dataframe['cti'] = pta.cti(dataframe['close'], length=20)
        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        # %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_96'] = williams_r(dataframe, period=96)
        # MAMA / FAMA
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.5, 0.05)
        dataframe['mama_diff'] = (dataframe['mama'] - dataframe['fama']) / dataframe['hl2']
        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] = (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3
        return dataframe
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if exit_reason == 'roi' and last_candle['min_l'] > last_candle['max_l'] * 3:
            return False
        return True
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # is_crash_1 = (
        #         (dataframe['tpct_change_1'] < 0.08) &
        #         (dataframe['tpct_change_2'] < 0.08) &
        #         (dataframe['tpct_change_4'] < 0.10)
        #     )
        # dataframe.loc[is_crash_1, 'enter_long'] = 1
        # dataframe.loc[is_crash_1, 'enter_tag'] = 'is crash'
        # Dip check
        # Bull confirm
        # Overpump check
        is_dip = (dataframe['close'] < dataframe['mama']) & (dataframe['r_14'] < -30) & (dataframe['cti'] < 3.0) & (dataframe['adx'] > 26) & (dataframe['mama_diff_1h'] > 0.003) & (dataframe['mama'] > dataframe['fama']) & (dataframe['sma_50'] > dataframe['sma_200'] * 1.01) & (dataframe['mama_1h'] > dataframe['fama_1h'] * 1.01) & (dataframe['rsi_84'] < 55) & (dataframe['rsi_112'] < 55) & (dataframe['cti_40_1h'] < 0.73) & (dataframe['r_96_1h'] < -6) & (dataframe['mama_diff_1h'] < 0.027) & (dataframe['close'].rolling(288).max() >= dataframe['close'] * 1.03)
        dataframe.loc[is_dip, 'enter_long'] = 1
        dataframe.loc[is_dip, 'enter_tag'] = 'is dip'
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_bb = (dataframe['close'] > dataframe['bb_upperband'] * 0.999) & (dataframe['rsi'] > 70)
        dataframe.loc[is_bb, 'exit_long'] = 1
        dataframe.loc[is_bb, 'exit_tag'] = 'BB upper'
        return dataframe
class StarRise_V2(StarRise_strat):
    INTERFACE_VERSION = 3
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #
    process_only_new_candles = True
    custom_info_trail_buy = dict()
    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800
    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = True
    trailing_expire_seconds_uptrend = 1800
    min_uptrend_trailing_profit = 0.005
    debug_mode = True
    trailing_buy_max_stop = 0.008  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.01  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))
    init_trailing_dict = {'trailing_buy_order_started': False, 'trailing_buy_order_uplimit': 0, 'start_trailing_price': 0, 'enter_tag': None, 'start_trailing_time': None, 'offset': 0, 'allow_trailing': False}

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
            duration = current_time - trailing_buy['start_trailing_time']
        except TypeError:
            duration = 0
        finally:
            logger.info(f"pair: {pair} : start: {trailing_buy['start_trailing_price']:.4f}, duration: {duration}, current: {current_price:.4f}, uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, profit: {self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}%, offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005
        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset
        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if current_trailing_profit_ratio > 0 and last_candle['enter_long'] == 1:
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif self.trailing_buy_uptrend_enabled and trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend and (current_trailing_profit_ratio < -1 * self.min_uptrend_trailing_profit):
            # less than 90s and price is rising, buy
            return 'forcebuy'
        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset
        trailing_buy_offset = {0.06: 0.02, 0.03: 0.01, 0: default_offset}
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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if len(dataframe) >= 1:
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)
                    if trailing_buy['allow_trailing']:
                        if not trailing_buy['trailing_buy_order_started'] and last_candle['enter_long'] == 1:
                            # start trailing buy
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_started'] = True
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_price'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag'] = f"initial_buy_tag (strat trail price {last_candle['close']})"
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_time'] = datetime.now(timezone.utc)
                            # self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = 0
                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['enter_tag'] = last_candle['enter_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f"start trailing buy for {pair} at {last_candle['close']}")
                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = '%.2f' % (self.current_trailing_profit_ratio(pair, current_price) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full')
                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')
                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy['trailing_buy_order_uplimit']
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit']}")
                            elif current_price < trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = '%.2f' % (self.current_trailing_profit_ratio(pair, current_price) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")
                            elif current_price > trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')
                    else:
                        logger.info(f'Wait for next buy signal for {pair}')
                if val == True:
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')
        return val

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if last_candle['enter_long'] == 1:
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True)]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['enter_tag'] if 'enter_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'enter_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            elif trailing_buy['trailing_buy_order_started'] == True:
                logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                dataframe.loc[:, 'enter_long'] = 1
                dataframe.loc[:, 'enter_tag'] = trailing_buy['enter_tag']
        # dataframe['buy'] = 1
        return dataframe
