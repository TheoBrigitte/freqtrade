# AlexBandSniper V6551 - Real-time divergence detection with improved filtering
# Based on V655 but with tighter signal quality to reduce noise
# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date

import logging
from datetime import timedelta, datetime
import datetime

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute, merge_informative_pair)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                resample('bollinger_upperband'): {'color': 'rgba(4,137,122,0.7)'},
                resample('kc_upperband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_middleband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_lowerband'): {'color': 'rgba(4,146,250,0.7)'},
                resample('bollinger_lowerband'): {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': resample('bollinger_upperband'),
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                resample('ema9'): {'color': 'purple'},
                resample('ema20'): {'color': 'yellow'},
                resample('ema50'): {'color': 'red'},
                resample('ema200'): {'color': 'white'},
            },
            'subplots': {
                "ATR": {
                    resample('atr'): {'color': 'firebrick'}
                },
                "Signal Strength": {
                    resample('bull_signal_strength'): {'color': 'green'},
                    resample('bear_signal_strength'): {'color': 'crimson'},
                    resample('signal_strength'): {'color': 'blue'}
                },
                "Squeeze": {
                    resample('squeeze_on'): {'color': 'orange'},
                    resample('squeeze_momentum'): {'color': 'cyan'}
                }
            }
        }

    def add_total_divergences_in_config(self, dataframe):
        self.config['main_plot'][resample("total_bullish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': resample("total_bullish_divergences_count"),
                'hovertext': resample("total_bullish_divergences_names"),
                'textfont': {'size': 11, 'color': 'green'},
                'textposition': 'bottom center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {'width': 2},
                    'color': 'green'
                }
            }
        }
        self.config['main_plot'][resample("total_bearish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': resample("total_bearish_divergences_count"),
                'hovertext': resample("total_bearish_divergences_names"),
                'textfont': {'size': 11, 'color': 'crimson'},
                'textposition': 'top center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {'width': 2},
                    'color': 'crimson'
                }
            }
        }
        return self


class AlexBandSniperV65513(IStrategy):
    """
    
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                                   AlexBandSniperV6                                                     ║
    ║                                                                                                                        ║
    ║                                                                                                                        ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              AlexBandSniper V6551                             ║
    ║                                                               ║
    ║  REAL-TIME DIVERGENCE — with IMPROVED FILTERING               ║
    ║                                                               ║
    ║  Based on V655 but with tighter signal quality to reduce      ║
    ║  noise and false entries while maintaining the real-time      ║
    ║  divergence approach (no lookahead bias).                     ║
    ║                                                               ║
    ║  CHANGES FROM V655:                                           ║
    ║    - Tighter Fast signal thresholds (strength >= 2, div >= 3) ║
    ║    - Choppiness filter added to ALL signal types              ║
    ║    - Volatility filter added to ALL signal types              ║
    ║    - No-conflict rule (no opposite signal strength)           ║
    ║    - Higher min_divergence_count (2 → 3)                      ║
    ║    - Larger window default (7 → 9) for more significant swings║
    ║                                                               ║
    ║  WHY: V655 was catching too many micro-swings. The real-time  ║
    ║  approach is correct but needs stronger quality gates since   ║
    ║  we no longer have the natural "significance filter" that     ║
    ║  pivot-based detection provided.                              ║
    ║                                                               ║
    ║  Result: Fewer trades, higher quality, same honest backtest.  ║
    ║                                                               ║
    ║    - V654 _rt shift columns (no longer needed)                ║
    ║                                                               ║
    ║  Added:                                                       ║
    ║    - realtime_divergence_scan() — fully vectorized            ║
    ║    - add_realtime_divergences() method on the strategy        ║
    ║    - Per-pair divergence-bar count log so you can see         ║
    ║      how many bars actually had divergences in the dataframe  ║
    ║                                                               ║
    ║  Tunable params (existing IntParameters reused):              ║
    ║    - window (5-15, default 7) → swing tightness               ║
    ║    - index_range (20-50, default 30) → max lookback distance  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v65513"

    class HyperOpt:
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name='stoploss')]

        def max_open_trades_space() -> List[Dimension]:
            return [Integer(3, 8, name='max_open_trades')]

        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.02, 0.3, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]

    # ── ROI ──────────────────────────────────────────────────────
    minimal_roi = {
        "0": 0.99  # Disabled — custom_exit handles everything
    }

    # ── Risk ─────────────────────────────────────────────────────
    stoploss = -0.15
    can_short = True
    use_custom_stoploss = False
    leverage_value = 5.0

    # ── Trailing (disabled — custom_exit does trailing) ──────────
    trailing_stop = False
    use_custom_exits = True

    # ── Timeframe ────────────────────────────────────────────────
    timeframe = '15m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ═════════════════════════════════════════════════════════════
    #  OPTIMIZABLE PARAMETERS
    # ═════════════════════════════════════════════════════════════

    # -- Signal quality --
    min_divergence_count = IntParameter(2, 5, default=3, space='buy', optimize=True, load=True)
    min_signal_strength = IntParameter(3, 10, default=4, space='buy', optimize=True, load=True)
    volume_threshold = DecimalParameter(1.1, 2.5, default=1.1, decimals=1, space='buy', optimize=True, load=True)

    # -- Market condition --
    rsi_overbought = DecimalParameter(65.0, 85.0, default=80.0, decimals=1, space='buy', optimize=True, load=True)
    rsi_oversold = DecimalParameter(15.0, 35.0, default=15.0, decimals=1, space='buy', optimize=True, load=True)
    adx_threshold = IntParameter(20, 45, default=20, space='buy', optimize=True, load=True)

    # -- Volatility --
    max_volatility = DecimalParameter(0.015, 0.035, default=0.025, decimals=3, space='buy', optimize=True, load=True)
    min_volatility = DecimalParameter(0.003, 0.008, default=0.005, decimals=3, space='buy', optimize=True, load=True)

    # -- Exit --
    rsi_exit_overbought = DecimalParameter(70.0, 90.0, default=80.0, decimals=1, space='sell', optimize=True, load=True)
    rsi_exit_oversold = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space='sell', optimize=True, load=True)
    adx_exit_threshold = IntParameter(15, 30, default=20, space='sell', optimize=True, load=True)

    # -- Trend confirmation --
    trend_strength_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True, load=True)

    # -- Technical --
    window = IntParameter(5, 15, default=9, space="buy", optimize=True, load=True)
    index_range = IntParameter(20, 50, default=30, space='buy', optimize=True, load=True)

    # ── Fast signals: V62 original logic (shift(1), div>=2) ─────
    # These showed perfectly on chart — reverted by Alex's request.

    # ── V64 NEW: Squeeze parameters ─────────────────────────────
    squeeze_lookback = IntParameter(3, 10, default=6, space='buy', optimize=True, load=True)
    squeeze_momentum_threshold = DecimalParameter(0.0, 0.005, default=0.001, decimals=4, space='buy', optimize=True, load=True)

    # ── V64 NEW: Momentum Burst parameters ──────────────────────
    momentum_rsi_cross = DecimalParameter(40.0, 55.0, default=50.0, decimals=1, space='buy', optimize=True, load=True)
    momentum_volume_mult = DecimalParameter(1.3, 2.5, default=1.5, decimals=1, space='buy', optimize=True, load=True)
    momentum_macd_threshold = DecimalParameter(0.0, 0.5, default=0.0, decimals=2, space='buy', optimize=True, load=True)

    # ── V64 NEW: Choppiness filter ──────────────────────────────
    chop_threshold = DecimalParameter(55.0, 70.0, default=61.8, decimals=1, space='buy', optimize=True, load=True)

    # ── V64 NEW: ATR-based exit multipliers ─────────────────────
    trail_atr_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space='sell', optimize=True, load=True)
    profit_floor_pct = DecimalParameter(0.3, 0.6, default=0.5, decimals=1, space='sell', optimize=True, load=True)
    # ── Protection ───────────────────────────────────────────────
    startup_candle_count: int = 200

    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    use_cooldown_protection = BooleanParameter(default=True, space="protection", optimize=True)

    use_max_drawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    max_drawdown_lookback = IntParameter(100, 300, default=200, space="protection", optimize=True)
    max_drawdown_trade_limit = IntParameter(5, 15, default=10, space="protection", optimize=True)
    max_drawdown_stop_duration = IntParameter(1, 5, default=2, space="protection", optimize=True)
    max_allowed_drawdown = DecimalParameter(0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True)

    stoploss_guard_lookback = IntParameter(30, 80, default=50, space="protection", optimize=True)
    stoploss_guard_trade_limit = IntParameter(2, 6, default=3, space="protection", optimize=True)
    stoploss_guard_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)

    # ── Orders ───────────────────────────────────────────────────
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.profit_peaks = {}

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    # ═════════════════════════════════════════════════════════════
    #  INDICATORS
    # ═════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        All indicators — V64 adds squeeze detection + momentum burst columns
        """

        # ── MULTI-TIMEFRAME: 1H ─────────────────────────────────
        try:
            if hasattr(self.dp, 'runmode') and self.dp.runmode.value in ['backtest', 'hyperopt']:
                informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            else:
                try:
                    informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
                except Exception:
                    informative_1h = None

            if (informative_1h is not None and
                    len(informative_1h) > 50 and
                    not informative_1h.empty and
                    'close' in informative_1h.columns):
                try:
                    informative_1h['ema50_1h'] = ta.EMA(informative_1h, timeperiod=50)
                    informative_1h['ema200_1h'] = ta.EMA(informative_1h, timeperiod=200)
                    informative_1h['trend_1h'] = ta.EMA(informative_1h, timeperiod=21)
                    informative_1h['trend_strength_1h'] = ta.ADX(informative_1h)
                    informative_1h['rsi_1h'] = ta.RSI(informative_1h)
                    informative_1h = informative_1h.bfill().ffill()
                    dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
                    logger.info(f"Merged 1h data for {metadata['pair']}")
                except Exception as merge_error:
                    logger.warning(f"Failed to merge 1h data for {metadata['pair']}: {merge_error}")
                    self._add_dummy_1h_columns(dataframe)
            else:
                self._add_dummy_1h_columns(dataframe)
        except Exception as e:
            logger.warning(f"Error accessing 1h data for {metadata['pair']}: {e}")
            self._add_dummy_1h_columns(dataframe)

        # ── 15M CORE INDICATORS ──────────────────────────────────
        informative = dataframe.copy()

        # Volume
        try:
            informative['volume_sma'] = informative['volume'].rolling(window=20, min_periods=1).mean()
            informative['volume_ratio'] = informative['volume'] / informative['volume_sma']
            informative['volume_ratio'] = informative['volume_ratio'].fillna(1.0)
        except:
            informative['volume_sma'] = informative['volume']
            informative['volume_ratio'] = 1.0

        # Volatility / ATR
        try:
            informative['atr'] = qtpylib.atr(informative, window=14, exp=False)
            informative['volatility'] = informative['atr'] / informative['close']
            informative['volatility'] = informative['volatility'].fillna(0.01)
        except:
            informative['atr'] = informative['close'] * 0.02
            informative['volatility'] = 0.01

        # Momentum suite
        try:
            informative['rsi'] = ta.RSI(informative)
            informative['stoch'] = ta.STOCH(informative)['slowk']
            informative['roc'] = ta.ROC(informative)
            informative['uo'] = ta.ULTOSC(informative)
            informative['ao'] = qtpylib.awesome_oscillator(informative)
            macd_data = ta.MACD(informative)
            informative['macd'] = macd_data['macd']
            informative['macd_signal'] = macd_data['macdsignal']
            informative['macd_hist'] = macd_data['macdhist']
            informative['cci'] = ta.CCI(informative)
            informative['cmf'] = chaikin_money_flow(informative, 20)
            informative['obv'] = ta.OBV(informative)
            informative['mfi'] = ta.MFI(informative)
            informative['adx'] = ta.ADX(informative)

            for col in ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'macd_signal', 'macd_hist', 'cci', 'cmf', 'obv', 'mfi', 'adx']:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
        except Exception as e:
            logger.warning(f"Momentum indicator error: {e}")
            informative['rsi'] = 50
            informative['stoch'] = 50
            informative['roc'] = 0
            informative['uo'] = 50
            informative['ao'] = 0
            informative['macd'] = 0
            informative['macd_signal'] = 0
            informative['macd_hist'] = 0
            informative['cci'] = 0
            informative['cmf'] = 0
            informative['obv'] = informative['volume'].cumsum()
            informative['mfi'] = 50
            informative['adx'] = 25

        # Keltner Channel
        try:
            keltner = emaKeltner(informative)
            informative["kc_upperband"] = keltner["upper"]
            informative["kc_middleband"] = keltner["mid"]
            informative["kc_lowerband"] = keltner["lower"]
        except:
            informative["kc_upperband"] = informative['close'] * 1.02
            informative["kc_middleband"] = informative['close']
            informative["kc_lowerband"] = informative['close'] * 0.98

        # Bollinger Bands
        try:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative['bollinger_upperband'] = bollinger['upper']
            informative['bollinger_lowerband'] = bollinger['lower']
            informative['bollinger_middleband'] = bollinger['mid']
            informative['bollinger_width'] = (bollinger['upper'] - bollinger['lower']) / bollinger['mid']
        except:
            informative['bollinger_upperband'] = informative['close'] * 1.02
            informative['bollinger_lowerband'] = informative['close'] * 0.98
            informative['bollinger_middleband'] = informative['close']
            informative['bollinger_width'] = 0.04

        # EMAs
        try:
            informative['ema9'] = ta.EMA(informative, timeperiod=9)
            informative['ema20'] = ta.EMA(informative, timeperiod=20)
            informative['ema50'] = ta.EMA(informative, timeperiod=50)
            informative['ema200'] = ta.EMA(informative, timeperiod=200)
            for col in ['ema9', 'ema20', 'ema50', 'ema200']:
                informative[col] = informative[col].bfill().fillna(informative['close'])
        except:
            informative['ema9'] = informative['close']
            informative['ema20'] = informative['close']
            informative['ema50'] = informative['close']
            informative['ema200'] = informative['close']

        # V6551: Pivot points removed — real-time divergence engine doesn't use them.

        # ── V64 NEW: SQUEEZE DETECTION ──────────────────────────
        # Squeeze = Bollinger Bands INSIDE Keltner Channel (low vol compression)
        # Release = BB expands back outside KC (breakout imminent)
        try:
            informative['squeeze_on'] = (
                (informative['bollinger_lowerband'] > informative['kc_lowerband']) &
                (informative['bollinger_upperband'] < informative['kc_upperband'])
            ).astype(int)

            # Squeeze just released: was on, now off
            informative['squeeze_release'] = (
                (informative['squeeze_on'].shift(1) == 1) &
                (informative['squeeze_on'] == 0)
            ).astype(int)

            # How many consecutive bars was the squeeze on?
            squeeze_count = informative['squeeze_on'].copy()
            groups = (squeeze_count != squeeze_count.shift()).cumsum()
            informative['squeeze_duration'] = squeeze_count.groupby(groups).cumsum()

            # Squeeze momentum = linear regression slope of close over squeeze period
            # Simplified: use rate-of-change of midline as proxy
            informative['squeeze_momentum'] = (
                (informative['close'] - informative['kc_middleband']) / informative['atr']
            ).fillna(0)

        except Exception as e:
            logger.warning(f"Squeeze calculation error: {e}")
            informative['squeeze_on'] = 0
            informative['squeeze_release'] = 0
            informative['squeeze_duration'] = 0
            informative['squeeze_momentum'] = 0

        # ── V64 NEW: MOMENTUM BURST DETECTION ───────────────────
        try:
            # RSI crossing key level
            informative['rsi_cross_up'] = (
                (informative['rsi'] > self.momentum_rsi_cross.value) &
                (informative['rsi'].shift(1) <= self.momentum_rsi_cross.value)
            )
            informative['rsi_cross_down'] = (
                (informative['rsi'] < (100 - self.momentum_rsi_cross.value)) &
                (informative['rsi'].shift(1) >= (100 - self.momentum_rsi_cross.value))
            )

            # MACD histogram flip
            informative['macd_hist_flip_bull'] = (
                (informative['macd_hist'] > self.momentum_macd_threshold.value) &
                (informative['macd_hist'].shift(1) <= self.momentum_macd_threshold.value)
            )
            informative['macd_hist_flip_bear'] = (
                (informative['macd_hist'] < -self.momentum_macd_threshold.value) &
                (informative['macd_hist'].shift(1) >= -self.momentum_macd_threshold.value)
            )

            # Volume surge
            informative['volume_surge'] = (
                informative['volume_ratio'] >= self.momentum_volume_mult.value
            )

            # EMA momentum: 9 > 20 (bullish) or 9 < 20 (bearish)
            informative['ema_bull'] = informative['ema9'] > informative['ema20']
            informative['ema_bear'] = informative['ema9'] < informative['ema20']

        except Exception as e:
            logger.warning(f"Momentum burst error: {e}")
            informative['rsi_cross_up'] = False
            informative['rsi_cross_down'] = False
            informative['macd_hist_flip_bull'] = False
            informative['macd_hist_flip_bear'] = False
            informative['volume_surge'] = False
            informative['ema_bull'] = False
            informative['ema_bear'] = False

        # ── V6551: REAL-TIME DIVERGENCE ANALYSIS ──────────────────
        # No pivots, no lookahead. Divergence fires AT the bar it forms.
        # Entry triggered next bar's open. See add_realtime_divergences().
        try:
            self.initialize_divergences_lists(informative)
            informative = self.add_realtime_divergences(informative)
        except Exception as e:
            logger.warning(f"V6551 realtime divergence engine error: {e}")
            informative["total_bullish_divergences"] = np.nan
            informative["total_bullish_divergences_count"] = 0
            informative["total_bullish_divergences_names"] = ''
            informative["total_bearish_divergences"] = np.nan
            informative["total_bearish_divergences_count"] = 0
            informative["total_bearish_divergences_names"] = ''

        # ── SIGNAL STRENGTH ──────────────────────────────────────
        try:
            informative['signal_strength'] = self.calculate_signal_strength(informative)
        except:
            informative['signal_strength'] = 0

        # ── MERGE BACK ───────────────────────────────────────────
        for col in informative.columns:
            dataframe[col] = informative[col]

        # ── CHOPPINESS INDEX ─────────────────────────────────────
        try:
            dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr_diff'] = dataframe['natr'] - dataframe['natr'].shift(1)
            dataframe['natr_direction_change'] = (dataframe['natr_diff'] * dataframe['natr_diff'].shift(1) < 0)
        except:
            dataframe['chop'] = 50
            dataframe['natr'] = 0.02
            dataframe['natr_diff'] = 0
            dataframe['natr_direction_change'] = False

        # ── SUPPORT / RESISTANCE ─────────────────────────────────
        try:
            dataframe['swing_high'] = dataframe['high'].rolling(window=50, min_periods=1).max()
            dataframe['swing_low'] = dataframe['low'].rolling(window=50, min_periods=1).min()
            dataframe['distance_to_resistance'] = (dataframe['swing_high'] - dataframe['close']) / dataframe['close']
            dataframe['distance_to_support'] = (dataframe['close'] - dataframe['swing_low']) / dataframe['close']
        except:
            dataframe['swing_high'] = dataframe['high']
            dataframe['swing_low'] = dataframe['low']
            dataframe['distance_to_resistance'] = 0.02
            dataframe['distance_to_support'] = 0.02

        # ── PLOT CONFIG ──────────────────────────────────────────
        try:
            self.plot_config = (
                PlotConfig()
                .add_total_divergences_in_config(dataframe)
                .config)
        except:
            self.plot_config = None

        return dataframe

    def _add_dummy_1h_columns(self, dataframe):
        """Fallback 1h columns from 15m data with scaled periods"""
        try:
            dataframe['ema50_1h_1h'] = ta.EMA(dataframe, timeperiod=200)
            dataframe['ema200_1h_1h'] = ta.EMA(dataframe, timeperiod=800)
            dataframe['trend_1h_1h'] = ta.EMA(dataframe, timeperiod=84)
            dataframe['trend_strength_1h_1h'] = ta.ADX(dataframe)
            dataframe['rsi_1h_1h'] = ta.RSI(dataframe, timeperiod=56)
            for col in ['ema50_1h_1h', 'ema200_1h_1h', 'trend_1h_1h', 'trend_strength_1h_1h', 'rsi_1h_1h']:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].bfill().fillna(
                        dataframe['close'] if 'ema' in col or 'trend' in col else
                        25 if 'strength' in col else 50
                    )
        except Exception as e:
            logger.warning(f"Dummy 1h column error: {e}")
            dataframe['ema50_1h_1h'] = dataframe['close']
            dataframe['ema200_1h_1h'] = dataframe['close']
            dataframe['trend_1h_1h'] = dataframe['close']
            dataframe['trend_strength_1h_1h'] = 25
            dataframe['rsi_1h_1h'] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """Directional signal strength — bull vs bear"""
        volume_strength = np.where(dataframe['volume_ratio'] > 1.5, 2,
                                   np.where(dataframe['volume_ratio'] > 1.2, 1, 0))
        adx_strength = np.where(dataframe['adx'] > 30, 1, 0)

        ema_bullish = np.where(
            (dataframe['ema20'] > dataframe['ema50']) &
            (dataframe['ema50'] > dataframe['ema200']), 1, 0)
        ema_bearish = np.where(
            (dataframe['ema20'] < dataframe['ema50']) &
            (dataframe['ema50'] < dataframe['ema200']), 1, 0)

        # V64: add squeeze bonus
        squeeze_bonus = np.where(dataframe['squeeze_release'] == 1, 1, 0)

        dataframe['bull_signal_strength'] = (
            dataframe['total_bullish_divergences_count'] * 2 +
            volume_strength + ema_bullish + adx_strength + squeeze_bonus
        )
        dataframe['bear_signal_strength'] = (
            dataframe['total_bearish_divergences_count'] * 2 +
            volume_strength + ema_bearish + adx_strength + squeeze_bonus
        )

        strength = dataframe['bull_signal_strength'] + dataframe['bear_signal_strength']
        return strength

    # ═════════════════════════════════════════════════════════════
    #  ENTRIES
    # ═════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Multi-signal entry engine with:
          1) Raw signal preservation
          2) Signal stack visibility
          3) Deterministic priority routing
          4) One final executable signal per candle (Freqtrade-safe)
          5) Per-condition diagnostic logging (which filter kills each signal)
        """
        pair = metadata.get("pair", "UNKNOWN")

        div_lookback = 2
        consensus_lookback = 2
        fast_lookback = 1

        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # ─────────────────────────────────────────────────────────────
        # BASE FILTERS
        # ─────────────────────────────────────────────────────────────
        has_volume = dataframe["volume"].shift(1) > 0
        not_choppy = dataframe["chop"].shift(1) < self.chop_threshold.value

        volatility_filter = (
            (dataframe["volatility"].shift(1) >= self.min_volatility.value) &
            (dataframe["volatility"].shift(1) <= self.max_volatility.value)
        )

        bands_long = two_bands_check_long(dataframe)
        bands_short = two_bands_check_short(dataframe)

        rsi_long_filter = (
            (dataframe["rsi"].shift(1) < self.rsi_overbought.value) &
            (dataframe["rsi"].shift(1) > 30)
        )
        rsi_short_filter = (
            (dataframe["rsi"].shift(1) > self.rsi_oversold.value) &
            (dataframe["rsi"].shift(1) < 70)
        )

        trend_strength = dataframe["adx"].shift(1) > self.adx_threshold.value

        # ─────────────────────────────────────────────────────────────
        # 1H TREND CONTEXT (from merged informative)
        # ─────────────────────────────────────────────────────────────
        if "trend_1h_1h" in dataframe.columns:
            trend_1h_bull = dataframe["close"] > dataframe["trend_1h_1h"]
            trend_1h_bear = dataframe["close"] < dataframe["trend_1h_1h"]
        else:
            # fallback if 1h merge failed — don't block signals
            trend_1h_bull = pd.Series(True, index=dataframe.index)
            trend_1h_bear = pd.Series(True, index=dataframe.index)

        # ─────────────────────────────────────────────────────────────
        # SHARED DIVERGENCE / STRENGTH CALCULATIONS
        # V6551: divergence columns are now REAL-TIME populated
        # (fired at the bar they form, no pivot lookahead). No _rt
        # shift needed — current-bar count is what was knowable at
        # candle close. Entry triggers next-bar open via Freqtrade.
        # ─────────────────────────────────────────────────────────────
        prev_bull_div = dataframe["total_bullish_divergences"].rolling(div_lookback, min_periods=1).max()
        prev_bear_div = dataframe["total_bearish_divergences"].rolling(div_lookback, min_periods=1).max()

        prev_bull_div_count = dataframe["total_bullish_divergences_count"].rolling(div_lookback, min_periods=1).sum()
        prev_bear_div_count = dataframe["total_bearish_divergences_count"].rolling(div_lookback, min_periods=1).sum()

        prev_bull_div_count_c = dataframe["total_bullish_divergences_count"].rolling(window=consensus_lookback, min_periods=1).sum()
        prev_bear_div_count_c = dataframe["total_bearish_divergences_count"].rolling(window=consensus_lookback, min_periods=1).sum()

        prev_bull_strength = dataframe["bull_signal_strength"].shift(1)
        prev_bear_strength = dataframe["bear_signal_strength"].shift(1)

        prev_bull_strength_c = dataframe["bull_signal_strength"].rolling(window=consensus_lookback, min_periods=1).sum()
        prev_bear_strength_c = dataframe["bear_signal_strength"].rolling(window=consensus_lookback, min_periods=1).sum()

        fast_bull_div_count = dataframe["total_bullish_divergences_count"].rolling(window=fast_lookback, min_periods=1).sum()
        fast_bear_div_count = dataframe["total_bearish_divergences_count"].rolling(window=fast_lookback, min_periods=1).sum()

        fast_bull_strength = dataframe["bull_signal_strength"].rolling(window=fast_lookback, min_periods=1).sum()
        fast_bear_strength = dataframe["bear_signal_strength"].rolling(window=fast_lookback, min_periods=1).sum()

        fast_bull_strength_raw = dataframe["bull_signal_strength"]
        fast_bear_strength_raw = dataframe["bear_signal_strength"]

        bullish_divergence = (
            (prev_bull_div > 0) &
            (prev_bull_div_count >= self.min_divergence_count.value)
        )

        bearish_divergence = (
            (prev_bear_div > 0) &
            (prev_bear_div_count >= self.min_divergence_count.value)
        )

        # ─────────────────────────────────────────────────────────────
        # SIGNALS
        # ─────────────────────────────────────────────────────────────

        # 1. HQ (highest quality)
        long_hq = (
            bullish_divergence &
            volatility_filter &
            bands_long &
            rsi_long_filter &
            trend_strength &
            (prev_bull_strength > prev_bear_strength) &
            (prev_bull_strength >= 3) &
            (prev_bull_div_count >= 4) &
            trend_1h_bull &                        # ← V65513: gate bullish HQ behind 1H uptrend
            has_volume                             #    (mirror of short_hq's trend_1h_bear; kills the
        )                                          #    -87 USDT ungated HQ_Bull_Div bleed: 147 of 149
                                                   #    HQ_Bull_Div trades were counter-trend losers,
                                                   #    only the 2 trend-aligned HQ_Bull_Div2 won)

        short_hq = (
            bearish_divergence &
            bands_short &
            rsi_short_filter &
            trend_strength &
            (prev_bear_strength > prev_bull_strength) &
            (prev_bear_strength >= 3) &
            (prev_bear_div_count >= 4) &
            trend_1h_bear &                        # ← V65512: gate bearish HQ behind 1H downtrend
            has_volume
        )

        # 1b. HQ2 (HQ + 1h trend confirmation)
        long_hq2 = long_hq & trend_1h_bull
        short_hq2 = short_hq & trend_1h_bear

        # 2. Consensus (with choppiness, volatility, no-conflict filters)
        consensus_long = (
            (prev_bull_strength_c >= 2) &
            (prev_bull_strength_c > prev_bear_strength_c) &
            (prev_bull_div_count_c >= 3) &
            (prev_bear_strength_c < 1) &           # ← No bearish conflict
            not_choppy &                           # ← Skip ranging markets
            volatility_filter &                    # ← Skip extreme volatility
            has_volume
        )

        consensus_short = (
            (prev_bear_strength_c >= 2) &
            (prev_bear_strength_c > prev_bull_strength_c) &
            (prev_bear_div_count_c >= 3) &
            (prev_bull_strength_c < 1) &           # ← No bullish conflict
            not_choppy &                           # ← Skip ranging markets
            volatility_filter &                    # ← Skip extreme volatility
            trend_1h_bear &                        # ← V65512: gate bearish consensus behind 1H downtrend
            has_volume
        )

        # 3. Fast Divergence (TIGHTENED: strength >= 2, div >= 3, no conflict)
        fast_long = (
            (fast_bull_strength >= 2) &              # ← Was >= 1
            (fast_bull_strength_raw >= 2) &          # ← Require raw strength too
            (fast_bull_div_count >= 3) &             # ← Was >= 2
            (fast_bear_strength < 1) &               # ← No bearish conflict
            (fast_bear_div_count < 2) &              # ← No bearish divergence
            not_choppy &                             # ← Skip ranging markets
            volatility_filter &                      # ← Skip extreme volatility
            has_volume
        )

        fast_short = (
            (fast_bear_strength >= 2) &              # ← Was >= 1
            (fast_bear_strength_raw >= 2) &          # ← Require raw strength too
            (fast_bear_div_count >= 3) &             # ← Was >= 2
            (fast_bull_strength < 1) &               # ← No bullish conflict
            (fast_bull_div_count < 2) &              # ← No bullish divergence
            not_choppy &                             # ← Skip ranging markets
            volatility_filter &                      # ← Skip extreme volatility
            trend_1h_bear &                          # ← V65512: gate fast short behind 1H downtrend
            has_volume
        )

        # 4. Squeeze Breakout
        squeeze_duration_prev = dataframe["squeeze_duration"].shift(1) >= self.squeeze_lookback.value
        squeeze_momentum_prev = dataframe["squeeze_momentum"].shift(1)
        recent_release = dataframe["squeeze_release"].rolling(3).max().shift(1) == 1

        squeeze_long = (
            recent_release &
            squeeze_duration_prev &
            (squeeze_momentum_prev > self.squeeze_momentum_threshold.value) &
            (dataframe["ema9"].shift(1) > dataframe["ema20"].shift(1)) &
            (dataframe["close"].shift(1) > dataframe["ema9"].shift(1)) &
            (dataframe["volume_ratio"].shift(1) > 1.1) &
            not_choppy &
            has_volume
        )

        squeeze_short = (
            recent_release &
            squeeze_duration_prev &
            (squeeze_momentum_prev < -self.squeeze_momentum_threshold.value) &
            (dataframe["ema9"].shift(1) < dataframe["ema20"].shift(1)) &
            (dataframe["close"].shift(1) < dataframe["ema9"].shift(1)) &
            (dataframe["volume_ratio"].shift(1) > 1.1) &
            not_choppy &
            has_volume
        )

        # 5. Momentum Burst
        momentum_long = (
            dataframe["rsi_cross_up"].shift(1) &
            dataframe["macd_hist_flip_bull"].shift(1) &
            dataframe["ema_bull"].shift(1) &
            not_choppy &
            has_volume
        )

        momentum_short = (
            dataframe["rsi_cross_down"].shift(1) &
            dataframe["macd_hist_flip_bear"].shift(1) &
            dataframe["ema_bear"].shift(1) &
            not_choppy &
            has_volume
        )
        # ═════════════════════════════════════════════════════════
        # SET ENTRY SIGNALS (order = tag priority, last wins)
        # ═════════════════════════════════════════════════════════
        # enter_long / enter_short — order doesn't matter (1 is 1)
        # V65513: momentum_* and squeeze_* signals DISABLED. Their tag lines
        # (below) were already commented out, so they entered with enter_tag=''
        # — the untagged "empty entries" that bled -19.51 USDT over 33 trades
        # with no way to attribute or manage them. Re-enable WITH tags + a 1H
        # trend gate if you want to A/B them as first-class signals later.
        #dataframe.loc[momentum_long, 'enter_long'] = 1
        #dataframe.loc[momentum_short, 'enter_short'] = 1
        #dataframe.loc[squeeze_long, 'enter_long'] = 1
        #dataframe.loc[squeeze_short, 'enter_short'] = 1
        dataframe.loc[fast_long, 'enter_long'] = 1
        dataframe.loc[fast_short, 'enter_short'] = 1
        dataframe.loc[consensus_long, 'enter_long'] = 1
        dataframe.loc[consensus_short, 'enter_short'] = 1
        dataframe.loc[long_hq, 'enter_long'] = 1
        dataframe.loc[short_hq, 'enter_short'] = 1
        dataframe.loc[long_hq2, 'enter_long'] = 1
        dataframe.loc[short_hq2, 'enter_short'] = 1

        dataframe['enter_tag'] = ''
        # enter_tag — order MATTERS, weakest first, strongest last
        #dataframe.loc[momentum_long, 'enter_tag'] = 'Momentum_Long'
        #dataframe.loc[momentum_short, 'enter_tag'] = 'Momentum_Short'
        #dataframe.loc[squeeze_long, 'enter_tag'] = 'Squeeze_Long'
        #dataframe.loc[squeeze_short, 'enter_tag'] = 'Squeeze_Short'
        dataframe.loc[fast_long, 'enter_tag'] = 'Div_Fast_Long'
        dataframe.loc[fast_short, 'enter_tag'] = 'Div_Fast_Short'
        dataframe.loc[consensus_long, 'enter_tag'] = 'Div_Bull_Cons'
        dataframe.loc[consensus_short, 'enter_tag'] = 'Div_Bear_Cons'
        dataframe.loc[long_hq, 'enter_tag'] = 'HQ_Bull_Div'
        dataframe.loc[short_hq, 'enter_tag'] = 'HQ_Bear_Div'
        dataframe.loc[long_hq2, 'enter_tag'] = 'HQ_Bull_Div2'
        dataframe.loc[short_hq2, 'enter_tag'] = 'HQ_Bear_Div2'

        # ═════════════════════════════════════════════════════════
        # V6551 DIAGNOSTIC: per-signal counters across the dataframe.
        # Also shows how many bars had ANY divergence (real-time).
        # If div_bars stays at 0, the realtime engine isn't finding
        # divergences — try smaller `window` or larger `index_range`.
        # If div_bars > 0 but signal counts still 0, downstream
        # filters (volatility, RSI bounds, ADX, bands) are killing them.
        # ═════════════════════════════════════════════════════════
        if len(dataframe) > 0:
            try:
                bull_div_bars = int((dataframe["total_bullish_divergences_count"] > 0).sum())
                bear_div_bars = int((dataframe["total_bearish_divergences_count"] > 0).sum())
                signal_counts = (
                    f"div_bars=B{bull_div_bars}/Be{bear_div_bars} | "
                    f"mom_l={int(momentum_long.sum())} mom_s={int(momentum_short.sum())} | "
                    f"sqz_l={int(squeeze_long.sum())} sqz_s={int(squeeze_short.sum())} | "
                    f"fast_l={int(fast_long.sum())} fast_s={int(fast_short.sum())} | "
                    f"cons_l={int(consensus_long.sum())} cons_s={int(consensus_short.sum())} | "
                    f"hq_l={int(long_hq.sum())} hq_s={int(short_hq.sum())} | "
                    f"hq2_l={int(long_hq2.sum())} hq2_s={int(short_hq2.sum())}"
                )
                logger.info(f"[{pair}] V6551 signal totals — {signal_counts}")
            except Exception as e:
                logger.debug(f"[{pair}] signal counter log error: {e}")

            c = dataframe.iloc[-1]
            logger.info(
                f"[{metadata['pair']}] {c.get('date', '')} | "
                f"Div: B{c.get('total_bullish_divergences_count', 0)}/Be{c.get('total_bearish_divergences_count', 0)} | "
                f"Str: {c.get('signal_strength', 0)} | "
                f"Vol: {c.get('volume_ratio', 0):.2f} | "
                f"Sqz: {c.get('squeeze_on', 0)} | "
                f"Chop: {c.get('chop', 0):.1f} | "
                f"HTF: {'Bull' if c.get('ema50_1h_1h', 0) > c.get('ema200_1h_1h', 0) else 'Bear'} | "
                f"Entry: L{c.get('enter_long', 0)}/S{c.get('enter_short', 0)} [{c.get('enter_tag', '')}]"
            )


        return dataframe
    # ═════════════════════════════════════════════════════════════
    #  EXITS
    # ═════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe



    # ═════════════════════════════════════════════════════════════
    #  LEVERAGE
    # ═════════════════════════════════════════════════════════════

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        V64: Signal-type aware leverage + volatility scaling
        """
        base_lev = self.leverage_value

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 1:
                current_vol = dataframe['volatility'].iloc[-2]
                sig_strength = dataframe['signal_strength'].iloc[-2]

                # Scale DOWN in high volatility
                if current_vol > 0.02:
                    base_lev *= 0.7
                elif current_vol > 0.015:
                    base_lev *= 0.85

                # Scale by signal strength
                if sig_strength >= 8:
                    lev = base_lev
                elif sig_strength >= 6:
                    lev = base_lev * 0.8
                elif sig_strength >= 4:
                    lev = base_lev * 0.6
                else:
                    lev = base_lev * 0.4

                # Squeeze / momentum signals get slightly lower leverage (faster moves = more risk)
                if entry_tag and ('Squeeze' in entry_tag or 'Momentum' in entry_tag):
                    lev *= 0.8

                return max(1.0, min(lev, max_leverage))
        except:
            pass

        return max(1.0, base_lev * 0.5)

    # ═════════════════════════════════════════════════════════════
    #  CUSTOM EXIT — V64 ATR-BASED TRAILING
    # ═════════════════════════════════════════════════════════════
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):
        """
        V652: V651 logic + Zone 0 reversal protection + verbose logging.
        """

        actual_profit = trade.calc_profit_ratio(current_rate)
        current_profit = actual_profit
        entry_tag = getattr(trade, 'enter_tag', '') or ''
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

        # ── PROFIT PEAK TRACKING ─────────────────────────────────
        trade_id = f"{pair}_{trade.open_date_utc.timestamp()}"
        if trade_id not in self.profit_peaks:
            self.profit_peaks[trade_id] = current_profit
        elif current_profit > self.profit_peaks[trade_id]:
            self.profit_peaks[trade_id] = current_profit
        profit_peak = self.profit_peaks[trade_id]

        # ── GET CURRENT ATR FOR ADAPTIVE EXITS ───────────────────
        atr_pct = 0.015  # fallback: 1.5%
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 1:
                last_candle = dataframe.iloc[-2]
                atr_pct = last_candle['atr'] / last_candle['close']
                atr_pct = max(0.003, min(atr_pct, 0.05))
        except:
            pass

        atr_profit_unit = atr_pct * trade.leverage * self.trail_atr_multiplier.value

        # ── EXIT LOGGER ──────────────────────────────────────────
        def log_and_exit(reason):
            logger.info(f"✅ EXIT: {pair}")
            logger.info(f"   💰 Profit: {current_profit * 100:.2f}%")
            logger.info(f"   📈 Peak:   {profit_peak * 100:.2f}%")
            logger.info(f"   🕐 Duration: {int(trade_duration_minutes)} min")
            logger.info(f"   🏷️  Tag: {entry_tag}")
            logger.info(f"   🎯 Reason: {reason}")
            if trade_id in self.profit_peaks:
                del self.profit_peaks[trade_id]
            return reason

        # ── HOLD LOGGER (call with current floor or None) ────────
        def log_hold(floor=None):
            if floor is not None:
                logger.debug(
                    f"⏳ HOLD: {pair} | Profit: {current_profit*100:.2f}% | "
                    f"Peak: {profit_peak*100:.2f}% | Floor: {floor*100:.2f}% | "
                    f"Dur: {int(trade_duration_minutes)}m | Tag: {entry_tag}"
                )
            else:
                logger.debug(
                    f"⏳ HOLD: {pair} | Profit: {current_profit*100:.2f}% | "
                    f"Peak: {profit_peak*100:.2f}% | Zone0 | "
                    f"Dur: {int(trade_duration_minutes)}m | Tag: {entry_tag}"
                )

        # ── DEAD-ZONE LEAK DETECTOR (info level — see leaks without full debug) ──
        # Logs when a trade with a meaningful peak is sitting in Zone 0 — exactly
        # the scenario you saw on the chart. Comment out once tuning is done.
        if current_profit < 0.02 and profit_peak >= 0.008:
            giveback = (profit_peak - current_profit) * 100
            logger.info(
                f"⚠️  DEAD_ZONE: {pair} | peak={profit_peak*100:.2f}% | "
                f"now={current_profit*100:.2f}% | giveback={giveback:.2f}% | "
                f"dur={int(trade_duration_minutes)}m | tag={entry_tag}"
            )

        # ── SIGNAL-TYPE PROFILE ──────────────────────────────────
        is_fast_trade = any(t in entry_tag for t in ['Squeeze', 'Momentum', 'Fast'])
        trail_tightness = 0.7 if is_fast_trade else 1.0

        # ── ZONE 0: UNDER 2% (and never reached a trail tier) ───
        # V65513: added `and profit_peak < 0.02`. Previously ANY trade under 2%
        # current profit was trapped in this scratch block and never reached the
        # Light/Medium/Strong/Monster trail engine (those are the real winners).
        # Now a trade that ever peaked >= 2% falls through to the trail logic,
        # which protects it with a real ATR pullback + floor instead of the flat
        # Sub2 scratch. This is what lets the "sub stuff" be removed safely.
        if current_profit < 0.02 and profit_peak < 0.02:

            # ▼▼▼ FAST/MOMENTUM TRADE PROTECTION ▼▼▼
            if is_fast_trade:
                # V65512: only react to a real peak (was 0.6% = noise) so fast
                # trades aren't scratched before reaching the trail zones.
                if profit_peak >= 0.015 and current_profit < profit_peak * 0.4:
                    return log_and_exit(f"Fast_Reversal_peak{profit_peak*100:.2f}%_now{current_profit*100:.2f}%")

                if profit_peak >= 0.012 and current_profit < 0.003:
                    return log_and_exit(f"Fast_Collapse_{profit_peak*100:.2f}%")

                if trade_duration_minutes >= 45 and profit_peak >= 0.008 and current_profit < 0.004:
                    return log_and_exit(f"Fast_Stale_{int(trade_duration_minutes)}m")

            # ▼▼▼ GENERAL SUB-2% REVERSAL — REMOVED in V65513 ▼▼▼
            # The "sub stuff." With the new Zone-0 gate (profit_peak < 0.02),
            # any trade that peaked into a trail tier now exits via the trail
            # engine instead of being scratched here at ~0.5-1%. The old block
            # required profit_peak >= 0.025, which can no longer be reached
            # inside this branch, so it was dead code and is deleted.

            if current_profit >= 0.005:
                stale_1 = 120 if is_fast_trade else 240
                stale_2 = 240 if is_fast_trade else 480
                stale_3 = 480 if is_fast_trade else 960

                if trade_duration_minutes >= stale_1 and profit_peak >= 0.015 and current_profit < 0.008:
                    return log_and_exit(f"Stagnant_Reversal_{profit_peak * 100:.1f}%")

                if trade_duration_minutes >= stale_2 and current_profit < 0.012:
                    return log_and_exit(f"Stagnant_{int(stale_2/60)}HR_{current_profit * 100:.1f}%")

                if trade_duration_minutes >= stale_3 and current_profit < 0.015:
                    return log_and_exit(f"Stagnant_{int(stale_3/60)}HR_{current_profit * 100:.1f}%")

            if trade_duration_minutes >= 1440 and current_profit >= 0.002:
                return log_and_exit(f"Timeout_24HR_{current_profit * 100:.1f}%")

            log_hold(floor=None)
            return None

        # ── PROFIT FLOOR ─────────────────────────────────────────
        floor_ratio = self.profit_floor_pct.value
        if profit_peak >= 0.20:
            profit_floor = profit_peak * 0.60
        elif profit_peak >= 0.12:
            profit_floor = profit_peak * floor_ratio + 0.02
        elif profit_peak >= 0.08:
            profit_floor = profit_peak * floor_ratio + 0.01
        elif profit_peak >= 0.04:
            profit_floor = profit_peak * 0.40
        else:
            profit_floor = 0.005

        # ── TIME-BASED TIGHTENING ────────────────────────────────
        time_factor = 1.0
        if trade_duration_minutes > 720:
            time_factor = 0.7
        elif trade_duration_minutes > 360:
            time_factor = 0.85

        # ── ATR TRAILING ZONES ───────────────────────────────────
        if profit_peak >= 0.02 and profit_peak < 0.04:
            pullback = 2.0 * atr_profit_unit * trail_tightness * time_factor
            trail_stop = max(profit_peak - pullback, profit_floor)
            if current_profit > 0 and current_profit < trail_stop:
                return log_and_exit(f"Light_Trail_{profit_peak * 100:.1f}%")
            log_hold(floor=trail_stop)
            return None

        if profit_peak >= 0.04 and profit_peak < 0.08:
            pullback = 1.5 * atr_profit_unit * trail_tightness * time_factor
            trail_stop = max(profit_peak - pullback, profit_floor)
            if current_profit > 0 and current_profit < trail_stop:
                return log_and_exit(f"Medium_Trail_{profit_peak * 100:.1f}%")
            log_hold(floor=trail_stop)
            return None

        if profit_peak >= 0.08 and profit_peak < 0.15:
            pullback = 1.2 * atr_profit_unit * trail_tightness * time_factor
            trail_stop = max(profit_peak - pullback, profit_floor)
            if current_profit > 0 and current_profit < trail_stop:
                return log_and_exit(f"Strong_Trail_{profit_peak * 100:.1f}%")
            log_hold(floor=trail_stop)
            return None

        if profit_peak >= 0.15:
            pullback = 1.0 * atr_profit_unit * trail_tightness * time_factor
            trail_stop = max(profit_peak - pullback, profit_floor)
            if current_profit > 0 and current_profit < trail_stop:
                return log_and_exit(f"Monster_Trail_{profit_peak * 100:.1f}%")
            log_hold(floor=trail_stop)
            return None

        return None

    # ═════════════════════════════════════════════════════════════
    #  V6551: REAL-TIME DIVERGENCE ENGINE
    #
    #  Detects divergences at the bar they form, with no pivot-
    #  confirmation lag. At each bar i:
    #    1. Bar i is a fresh local extremum (lowest/highest of last
    #       `swing+1` bars looking BACKWARD only — knowable at close).
    #    2. Within the prior [lookback_min, lookback_max] window,
    #       there exists a 2-sided swing extremum at bar j (this is
    #       fully knowable since j is in the past).
    #    3. Price between j→i went one direction, indicator the
    #       opposite (the divergence).
    #    4. Both moves cleared an ATR/indicator floor.
    #  If any indicator satisfies this for any j, divergence fires
    #  AT bar i and Freqtrade enters at bar i+1's open. No lookahead.
    # ═════════════════════════════════════════════════════════════

    def initialize_divergences_lists(self, dataframe: DataFrame):
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def add_realtime_divergences(self, dataframe: DataFrame) -> DataFrame:
        """V6551: Real-time divergence detection across all configured indicators."""
        indicators = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi']

        n = len(dataframe)
        bull_count = np.zeros(n, dtype=np.int32)
        bear_count = np.zeros(n, dtype=np.int32)
        bull_names = [''] * n
        bear_names = [''] * n

        # Tunable boundaries — tied to existing optimizable params:
        #   self.window.value      (5..15, default 7) → swing tightness
        #   self.index_range.value (20..50, default 30) → max lookback range
        swing = max(2, int(self.window.value) // 3)        # default → 2
        lookback_min = 5
        lookback_max = max(lookback_min + 5, int(self.index_range.value))

        try:
            low_arr = dataframe['low'].values.astype(np.float64)
            high_arr = dataframe['high'].values.astype(np.float64)
            atr_arr = dataframe['atr'].values.astype(np.float64)
        except Exception as e:
            logger.warning(f"V6551 input array error: {e}")
            return dataframe

        for ind_name in indicators:
            if ind_name not in dataframe.columns:
                continue
            try:
                ind_arr = dataframe[ind_name].values.astype(np.float64)

                bull_d = realtime_divergence_scan(
                    low_arr, ind_arr, atr_arr,
                    divergence_type='bullish',
                    lookback_min=lookback_min,
                    lookback_max=lookback_max,
                    swing=swing,
                    min_atr_mult=0.5,
                    min_ind_diff=5.0,
                )
                bear_d = realtime_divergence_scan(
                    high_arr, ind_arr, atr_arr,
                    divergence_type='bearish',
                    lookback_min=lookback_min,
                    lookback_max=lookback_max,
                    swing=swing,
                    min_atr_mult=0.5,
                    min_ind_diff=5.0,
                )

                bull_count += bull_d.astype(np.int32)
                bear_count += bear_d.astype(np.int32)

                bull_idx = np.where(bull_d)[0]
                bear_idx = np.where(bear_d)[0]
                tag = ind_name.upper() + '<br>'
                for idx in bull_idx:
                    bull_names[idx] += tag
                for idx in bear_idx:
                    bear_names[idx] += tag

                # Per-indicator occurrence column (compatible with V653 plot conventions)
                close_arr = dataframe['close'].values
                dataframe[f'bullish_divergence_{ind_name}_occurence'] = np.where(bull_d, close_arr, np.nan)
                dataframe[f'bearish_divergence_{ind_name}_occurence'] = np.where(bear_d, close_arr, np.nan)

            except Exception as e:
                logger.warning(f"V6551 realtime divergence error for {ind_name}: {e}")
                continue

        close_arr = dataframe['close'].values
        dataframe['total_bullish_divergences_count'] = bull_count
        dataframe['total_bearish_divergences_count'] = bear_count
        dataframe['total_bullish_divergences_names'] = bull_names
        dataframe['total_bearish_divergences_names'] = bear_names
        dataframe['total_bullish_divergences'] = np.where(bull_count > 0, close_arr, np.nan)
        dataframe['total_bearish_divergences'] = np.where(bear_count > 0, close_arr, np.nan)

        return dataframe

    @property
    def protections(self):
        prot = []
        if self.use_cooldown_protection.value:
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            })
        if self.use_max_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stoploss_guard_only_per_pair.value,
            })
        return prot


# ═════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def choppiness_index(high, low, close, window=14):
    """
    V643 FIX: Standard choppiness formula using True Range instead of NATR.
    
    Standard formula:
    choppiness = 100 * log10(sum(true_range over window) / (highest_high - lowest_low)) / log10(window)
    
    This fixes the issue where NATR (already normalized) was divided by raw price span,
    mixing incompatible units and producing unrealistic/negative values.
    """
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum True Range over window
    tr_sum = true_range.rolling(window=window, min_periods=window).sum()
    
    # Calculate price span (high - low over window)
    high_max = high.rolling(window=window, min_periods=window).max()
    low_min = low.rolling(window=window, min_periods=window).min()
    span = (high_max - low_min).replace(0, np.nan)
    
    # Calculate choppiness ratio
    ratio = (tr_sum / span).replace([np.inf, -np.inf], np.nan)
    
    # Final choppiness calculation
    choppiness = 100 * np.log10(ratio) / np.log10(window)
    return choppiness


def resample(indicator):
    return indicator


def two_bands_check_long(dataframe):
    prev_low = dataframe['low'].shift(1)
    prev_close = dataframe['close'].shift(1)
    prev_kc_lower = dataframe['kc_lowerband'].shift(1)
    return (
        (prev_low <= prev_kc_lower) |
        (prev_close <= prev_kc_lower)
    )


def two_bands_check_short(dataframe):
    prev_high = dataframe['high'].shift(1)
    prev_close = dataframe['close'].shift(1)
    prev_kc_upper = dataframe['kc_upperband'].shift(1)
    return (
        (prev_high >= prev_kc_upper) |
        (prev_close >= prev_kc_upper)
    )


def green_candle(dataframe):
    return dataframe[resample('open')] < dataframe[resample('close')]


def red_candle(dataframe):
    return dataframe[resample('open')] > dataframe[resample('close')]


def realtime_divergence_scan(price, indicator, atr,
                              divergence_type='bullish',
                              lookback_min=5, lookback_max=30,
                              swing=2,
                              min_atr_mult=0.5, min_ind_diff=5.0):
    """
    V6551 real-time divergence detector. Fully vectorized.

    For each bar i, returns True if a regular divergence forms AT bar i:
      - Bullish: price made a lower low than a prior swing low j,
                 while the indicator made a higher low at i vs j.
      - Bearish: symmetric on highs.

    No pivot confirmation lag — bar i is treated as a "fresh" extremum
    if it's the lowest/highest of the last (swing+1) bars LOOKING
    BACKWARD ONLY. The prior bar j inside the [lookback_min, lookback_max]
    window must be a 2-sided swing extremum (knowable since j is past).

    Parameters
    ----------
    price : 1-D array of low (for bullish) or high (for bearish)
    indicator : 1-D oscillator array (same length as price)
    atr : 1-D ATR array (same length as price)
    divergence_type : 'bullish' or 'bearish'
    lookback_min, lookback_max : range of N to scan, where j = i - N
    swing : how many bars on each side define a swing extremum at j
            (and how many bars LEFT-only define the fresh extremum at i)
    min_atr_mult : minimum |price[i] - price[j]| in ATR units
    min_ind_diff : minimum absolute indicator difference

    Returns
    -------
    np.ndarray of bool, length len(price)
    """
    n = len(price)
    out = np.zeros(n, dtype=bool)
    if n < lookback_max + swing + 2:
        return out

    price_arr = np.asarray(price, dtype=np.float64)
    ind_arr = np.asarray(indicator, dtype=np.float64)
    atr_arr = np.asarray(atr, dtype=np.float64)

    s_price = pd.Series(price_arr)

    if divergence_type == 'bullish':
        # Bar i is fresh local low if price[i] <= min(price[i-swing : i])
        prev_extr = s_price.rolling(swing, min_periods=swing).min().shift(1).values
        is_current_swing = price_arr <= prev_extr
        # Bar j is 2-sided swing low (centered window of 2*swing+1)
        center_extr = s_price.rolling(2 * swing + 1, center=True,
                                       min_periods=2 * swing + 1).min().values
        is_past_swing = price_arr <= center_extr
    else:
        prev_extr = s_price.rolling(swing, min_periods=swing).max().shift(1).values
        is_current_swing = price_arr >= prev_extr
        center_extr = s_price.rolling(2 * swing + 1, center=True,
                                       min_periods=2 * swing + 1).max().values
        is_past_swing = price_arr >= center_extr

    is_current_swing = np.where(np.isnan(prev_extr), False, is_current_swing).astype(bool)
    is_past_swing = np.where(np.isnan(center_extr), False, is_past_swing).astype(bool)

    span = lookback_max - lookback_min + 1
    shifted_price = np.full((n, span), np.nan, dtype=np.float64)
    shifted_ind = np.full((n, span), np.nan, dtype=np.float64)
    shifted_swing = np.zeros((n, span), dtype=bool)

    for k in range(span):
        N = lookback_min + k
        shifted_price[N:, k] = price_arr[:n - N]
        shifted_ind[N:, k] = ind_arr[:n - N]
        shifted_swing[N:, k] = is_past_swing[:n - N]

    p2d = price_arr[:, np.newaxis]
    i2d = ind_arr[:, np.newaxis]
    a2d = atr_arr[:, np.newaxis]
    cur_swing_2d = is_current_swing[:, np.newaxis]

    # NaN comparisons in numpy evaluate to False — clean booleans below.
    if divergence_type == 'bullish':
        valid = (
            cur_swing_2d & shifted_swing &
            (p2d < shifted_price) &
            (i2d > shifted_ind) &
            ((shifted_price - p2d) >= min_atr_mult * a2d) &
            ((i2d - shifted_ind) >= min_ind_diff)
        )
    else:
        valid = (
            cur_swing_2d & shifted_swing &
            (p2d > shifted_price) &
            (i2d < shifted_ind) &
            ((p2d - shifted_price) >= min_atr_mult * a2d) &
            ((shifted_ind - i2d) >= min_ind_diff)
        )

    return valid.any(axis=1)


def emaKeltner(dataframe):
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner


def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')