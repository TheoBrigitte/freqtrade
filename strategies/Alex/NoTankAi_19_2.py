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


class NOTankAi_19_2(IStrategy):
    """
    NOTankAi_19_2 — LOOKAHEAD-BIAS FIX on top of v19_1.

    Changes vs v19_1:
      0. LOOKAHEAD-BIAS FIX (the big one)
         Root cause: `populate_indicators` used scipy.signal.argrelextrema
         with order=5 to label `maxima`, `minima`, and `&s-extrema`. That
         function compares each candle to its 5 neighbours on BOTH sides,
         so a candle is only labelled a max/min once 5 FUTURE candles have
         closed lower/higher. In backtest the whole dataframe exists, so
         every real top/bottom gets retroactively labelled perfectly and
         the strategy looks like it's nailing reversals. In live trading
         those 5 future candles literally don't exist yet on the most
         recent bars, so the label can never appear in time — by the time
         it lights up, the move has already run for at least 5 candles.
         The downstream effect was severe on shorts: in live we'd short
         a "local max" that was actually a tiny pullback inside a real
         uptrend (see the WIF -65% short), because the labeler that made
         the backtest look good wasn't available yet at decision time.
         Same bias on longs, but crypto's general drift, DCA, and the
         wide ROI grid mostly papered over it.

         Fix: shift the argrelextrema labels forward by `order` candles,
         so the marker only emits on the candle where its future-confirmation
         would actually be available in live (i.e. AFTER the 5 future
         candles have closed). This makes backtest behaviour match live
         behaviour. Expect significantly worse but HONEST backtest numbers.

         Affected columns: `maxima`, `minima`, `&s-extrema`, plus the
         downstream `maxima_check` / `minima_check` derived from them.

         Side note: ALL other rolling/expanding indicators in this file
         were audited and are causal (they only use past data). The bias
         was localised entirely to the argrelextrema block.

    Inherited from v19_1:
      1. Reversal exits live in `custom_exit` (bypass exit_profit_only).
      2. ADX+ROC strong-trend bailout in `custom_exit`.
      3. Profit-only pyramid-up DCA (never average down).
      4. Entry logic otherwise unchanged from v18 / v19_1.
    """

    exit_profit_only = True
    trailing_stop = False
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 1
    max_dca_multiplier = 1
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.15
    timeframe = "15m"

    # DCA
    position_adjustment_enable = False
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

    # ======================================================================
    # BTC-driven dynamic RSI (Pete Wong's core change vs vanilla NOTankAi)
    # ======================================================================

    def cal_btc_rsi(self):
        """Mean RSI of BTC over `btc_rsi_window_size` candles. Returns -1 on failure."""
        window_size = self.config.get("btc_rsi_window_size", 8)
        btc_data = self.dp.get_pair_dataframe('BTC/USDT:USDT', timeframe=self.timeframe)
        if btc_data is None or len(btc_data) < 14:
            logger.info(f'In cal_btc_rsi, btc_data len: {len(btc_data) if btc_data is not None else 0}')
            return -1
        btc_data["rsi"] = ta.RSI(btc_data)
        btc_data["rsi_mean"] = btc_data["rsi"].rolling(window=window_size).mean()
        last = btc_data.iloc[-1]["rsi_mean"]
        if pd.isna(last):
            return -1
        return last

    def dynamic_rsi(self, metadata: dict):
        """
        Returns (rsi_low_move, rsi_high_move) — offsets applied to the base
        long (30) and short (70) RSI thresholds based on BTC's mean RSI.

        Behaviour:
          - BTC bullish (RSI > normal_high)  -> raise both thresholds
              (more longs allowed, fewer shorts allowed -> ride the trend)
          - BTC bearish (RSI < normal_low)   -> lower both thresholds
              (fewer longs allowed, more shorts allowed)
          - BTC data unavailable             -> return (0, 0) -> base 30/70
        """
        normal_btc_rsi_high = self.config.get('normal_btc_rsi_high', 60.0)
        normal_btc_rsi_low = self.config.get('normal_btc_rsi_low', 40.0)
        btc_rsi_delta = self.config.get('btc_rsi_delta', 2.0)

        btc_rsi = self.cal_btc_rsi()

        # FIX vs v16: bail out cleanly when BTC data is unavailable.
        # In v16 a -1 sentinel propagated into the math and produced a
        # long threshold of ~9.5 and a short threshold of 60, which silently
        # killed all longs and loosened shorts during BTC data gaps.
        if btc_rsi is None or btc_rsi < 0 or pd.isna(btc_rsi):
            logger.warning(f"{metadata.get('pair', '?')} BTC RSI unavailable; using base 30/70.")
            return 0.0, 0.0

        logger.debug(f'In dynamic_rsi, got btc_rsi: {btc_rsi}.')

        rsi_low_move_origin = (btc_rsi - normal_btc_rsi_low) / btc_rsi_delta
        rsi_low_move_origin = min(20.0, rsi_low_move_origin)

        rsi_high_move = (btc_rsi - normal_btc_rsi_high) / btc_rsi_delta
        rsi_high_move = max(-10.0, rsi_high_move)

        pair = metadata['pair']
        rsi_low_move_final = rsi_low_move_origin

        if pair in self.config.get("decay_pair_map", {}):
            factor = max(1.0, (self.config.get("decay_pair_map").get(pair, 0.0) + 1.5))
            if rsi_low_move_origin > 0.0:
                rsi_low_move_final = rsi_low_move_origin / factor
            else:
                rsi_low_move_final = rsi_low_move_origin * factor

        logger.debug(
            f'In dynamic_rsi, pair: {pair}, rsi_low_move: {rsi_low_move_origin}, '
            f'rsi_low_move_final: {rsi_low_move_final}, rsi_high_move: {rsi_high_move}.'
        )
        return rsi_low_move_final, rsi_high_move

    # ======================================================================
    # Signal filters (from message.txt) — all vectorized with rolling ops
    # ======================================================================

    def check_volume_change_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect 1h volume spikes (4 x 15m candles aggregated)."""
        volume_times = self.config.get("volume_times", 3)
        volume_check_window = self.config.get("volume_check_window", 8)
        past_hours_window = 4  # 4 x 15m = 1h

        df['past_hour_volume'] = df['volume'].rolling(window=past_hours_window).sum()
        df['past_hours_avg_volume'] = (
            df['past_hour_volume'].rolling(window=volume_check_window).mean()
        )
        df['volume_change_hourly'] = (
            df['past_hour_volume'] > volume_times * df['past_hours_avg_volume']
        ).astype(int)
        df['volume_change_hourly_check'] = (
            df['volume_change_hourly']
            .rolling(window=(volume_check_window * past_hours_window))
            .max()
            .fillna(0)
            .astype(int)
        )
        return df

    def check_volume_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect short-term (single-candle) volume spikes."""
        volume_times = self.config.get("volume_times", 3)
        volume_check_window = self.config.get("volume_check_window", 8)

        df['volume_change'] = (
            df['volume'] > volume_times * df['volume'].rolling(window=4).mean().shift(1)
        ).astype(int)
        df['volume_change_check'] = (
            df['volume_change']
            .rolling(window=volume_check_window)
            .max()
            .fillna(0)
            .astype(int)
        )
        return df

    def check_upper(self, df: DataFrame) -> pd.DataFrame:
        """Flag candles in a strong sustained uptrend (vectorized)."""
        upper_period = self.config.get("upper_period", 8)
        check_upper_period = self.config.get("check_upper_period", 3)
        upper_ratio = self.config.get("upper_ratio", 0.875)
        threshold = upper_period * upper_ratio

        df['is_up'] = (df['close'] >= df['open']).astype(int)
        df['is_upper'] = (
            df['is_up'].rolling(window=upper_period).sum() >= threshold
        ).astype(int)
        df['check_upper'] = (
            df['is_upper'].rolling(window=check_upper_period).sum().fillna(0).astype(int)
        )
        return df

    def check_down(self, df: DataFrame) -> pd.DataFrame:
        """Flag candles in a strong sustained downtrend (vectorized)."""
        down_period = self.config.get("down_period", 8)
        check_down_period = self.config.get("check_down_period", 3)
        down_ratio = self.config.get("down_ratio", 0.875)
        threshold = down_period * down_ratio

        df['is_down'] = (df['close'] <= df['open']).astype(int)
        df['is_downing'] = (
            df['is_down'].rolling(window=down_period).sum() >= threshold
        ).astype(int)
        df['check_down'] = (
            df['is_downing'].rolling(window=check_down_period).sum().fillna(0).astype(int)
        )
        return df

    def is_continuous_up(self, df: DataFrame) -> pd.DataFrame:
        """1 if the last `continuous_check_window` candles all closed up."""
        check_window = self.config.get("continuous_check_window", 5)
        directions = (df['close'] >= df['open']).astype(int)
        df["continuous_up"] = (
            directions.rolling(window=check_window).sum() == check_window
        ).astype(int)
        return df

    def is_continuous_down(self, df: DataFrame) -> pd.DataFrame:
        """1 if the last `continuous_check_window` candles all closed down."""
        check_window = self.config.get("continuous_check_window", 5)
        directions = (df['close'] <= df['open']).astype(int)
        df["continuous_down"] = (
            directions.rolling(window=check_window).sum() == check_window
        ).astype(int)
        return df

    # ======================================================================
    # Trade management
    # ======================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        v19: Reversal + runaway-trend bailouts that bypass exit_profit_only.

        `exit_profit_only=True` filters exit_long/exit_short signals from
        populate_exit_trend when current_profit < 0. `custom_exit` is NOT
        subject to that filter, so this is where we put exits that MUST fire
        regardless of PnL — specifically the cases where the market has
        clearly turned against the position (the TON-style runaway short).

        Returns an exit_reason string (or None to do nothing).
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        last = df.iloc[-1]

        use_reversal = self.config.get("custom_exit_use_reversal", True)
        use_runaway = self.config.get("custom_exit_use_runaway", True)

        if trade.is_short:
            if use_reversal and last.get("enter_long", 0) == 1:
                logger.info(
                    f"{pair} custom_exit: closing short on reversal long signal "
                    f"(profit={current_profit:.3f})"
                )
                return "reversal_long_signal"
            if use_runaway and last.get("strong_uptrend", 0) == 1:
                logger.info(
                    f"{pair} custom_exit: closing short on strong_uptrend "
                    f"(profit={current_profit:.3f}, "
                    f"roc={last.get('price_roc', float('nan')):.2f}%, "
                    f"adx={last.get('adx', float('nan')):.1f})"
                )
                return "extreme_uptrend"
        else:
            if use_reversal and last.get("enter_short", 0) == 1:
                logger.info(
                    f"{pair} custom_exit: closing long on reversal short signal "
                    f"(profit={current_profit:.3f})"
                )
                return "reversal_short_signal"
            if use_runaway and last.get("strong_downtrend", 0) == 1:
                logger.info(
                    f"{pair} custom_exit: closing long on strong_downtrend "
                    f"(profit={current_profit:.3f}, "
                    f"roc={last.get('price_roc', float('nan')):.2f}%, "
                    f"adx={last.get('adx', float('nan')):.1f})"
                )
                return "extreme_downtrend"
        return None

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
        """
        v19: Profit-only DCA. Pyramid into winners; never average down.

        Sequence:
          1. Partial profit-taking at +25% / +40% (preserved from v18).
          2. Block all DCA when current_profit <= 0 (the user's main fix).
          3. Otherwise, only DCA inside a configured profit band
             (default +3% to +15%) so we don't pile in right at exit target.
        """
        # --- 1) Partial profit-taking (unchanged) ---
        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)

        # --- 2) NEVER add to a losing trade (was the v18 behaviour) ---
        if current_profit <= 0:
            return None

        # --- 3) Pyramid-up band: only DCA in modest-profit zone ---
        dca_profit_min = self.config.get("dca_profit_min", 0.03)   # +3%
        dca_profit_max = self.config.get("dca_profit_max", 0.15)   # +15%
        if current_profit < dca_profit_min:
            return None
        if current_profit > dca_profit_max:
            # too close to exit target — don't add late
            return None

        # Respect max_entry_position_adjustment
        if trade.nr_of_successful_entries > self.max_entry_position_adjustment:
            return None

        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            if not filled_entries:
                return None
            stake_amount = filled_entries[0].cost
            logger.info(
                f"{trade.pair} pyramid-up DCA: profit={current_profit:.3f} "
                f"entry#{trade.nr_of_successful_entries + 1} stake={stake_amount}"
            )
            return stake_amount
        except Exception as exception:
            logger.warning(f"{trade.pair} DCA stake calc failed: {exception}")
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
        base_leverage = 3

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

    # ======================================================================
    # Feature engineering (freqai-style; harmless when freqai is not used)
    # ======================================================================

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

    # ======================================================================
    # Indicators
    # ======================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        # ------------------------------------------------------------------
        # LOOKAHEAD-BIAS FIX (v19_2)
        # ------------------------------------------------------------------
        # scipy.signal.argrelextrema(..., order=N) labels a candle as a local
        # max/min by comparing it to N candles on BOTH sides. The label can
        # therefore only be set after N future candles have closed. In a
        # backtest that's free; in live trading those candles don't exist
        # yet on the most recent bars. The original v19_1 code used the raw
        # labels directly, which made backtests retroactively "nail" every
        # top/bottom that the strategy could not have seen in real time.
        #
        # Fix: emit the marker `extrema_order` candles AFTER the pivot index,
        # i.e. on the candle where the N future-candles needed to confirm
        # the pivot are guaranteed to have closed. Backtest now behaves like
        # live. Knock-on effect: maxima_check / minima_check (computed below
        # as rolling windows over `maxima` / `minima`) inherit the fix
        # automatically, no extra change needed.
        # ------------------------------------------------------------------
        extrema_order = self.config.get("extrema_order", 5)

        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=extrema_order)[0]
        min_peaks = argrelextrema(dataframe["close"].values, np.less,    order=extrema_order)[0]

        maxima_raw = pd.Series(0, index=dataframe.index, dtype=int)
        minima_raw = pd.Series(0, index=dataframe.index, dtype=int)
        if len(max_peaks) > 0:
            maxima_raw.iloc[max_peaks] = 1
        if len(min_peaks) > 0:
            minima_raw.iloc[min_peaks] = 1

        # Shift forward by `extrema_order` so labels only appear once their
        # future-confirmation candles have actually closed.
        dataframe["maxima"] = maxima_raw.shift(extrema_order).fillna(0).astype(int)
        dataframe["minima"] = minima_raw.shift(extrema_order).fillna(0).astype(int)

        extrema_raw = pd.Series(0.0, index=dataframe.index)
        if len(min_peaks) > 0:
            extrema_raw.iloc[min_peaks] = -1
        if len(max_peaks) > 0:
            extrema_raw.iloc[max_peaks] = 1
        dataframe["&s-extrema"] = extrema_raw.shift(extrema_order).fillna(0)

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

        # ---- Apply message.txt signal filters once, here, so they're cached ----
        dataframe = self.check_volume_change(dataframe)
        dataframe = self.check_volume_change_hourly(dataframe)
        dataframe = self.check_upper(dataframe)
        dataframe = self.check_down(dataframe)
        dataframe = self.is_continuous_up(dataframe)
        dataframe = self.is_continuous_down(dataframe)

        # ---- v19: Strong-trend detection (TON-style runaway moves) ----
        # ADX measures trend STRENGTH (not direction); ROC gives direction + magnitude.
        # Combining them catches one-sided pumps/dumps that the candle-color
        # `check_upper` / `check_down` rolling counts can miss when there are
        # mixed-color but net-strongly-trending candles.
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        roc_window = self.config.get("strong_trend_roc_window", 20)  # 20 * 15m = 5h
        dataframe["price_roc"] = dataframe["close"].pct_change(periods=roc_window) * 100

        strong_roc_pct = self.config.get("strong_trend_roc_pct", 8.0)   # 8% in 5h
        strong_adx = self.config.get("strong_trend_adx", 30.0)

        dataframe["strong_uptrend"] = (
            (dataframe["price_roc"] > strong_roc_pct) &
            (dataframe["adx"] > strong_adx)
        ).astype(int)
        dataframe["strong_downtrend"] = (
            (dataframe["price_roc"] < -strong_roc_pct) &
            (dataframe["adx"] > strong_adx)
        ).astype(int)

        # ---- Cache dynamic RSI thresholds on the dataframe ----
        # FIX vs v16: dynamic_rsi() was called twice per candle (entry + exit
        # trends), each time refetching BTC data. Compute it once here.
        rsi_low_adj, rsi_high_adj = self.dynamic_rsi(metadata)
        dataframe["rsi_low_adj"] = rsi_low_adj
        dataframe["rsi_high_adj"] = rsi_high_adj

        # v19: Cache BTC mean RSI for the short-block gate so we don't
        # refetch it inside populate_entry_trend.
        btc_rsi_value = self.cal_btc_rsi()
        if btc_rsi_value is None or btc_rsi_value < 0 or pd.isna(btc_rsi_value):
            btc_rsi_value = -1.0  # sentinel: BTC data unavailable
        dataframe["btc_rsi_mean"] = btc_rsi_value

        pair = metadata["pair"]
        if dataframe["maxima"].iloc[-3] == 1 and dataframe["maxima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Maxima Detected - Potential Short!!!")
        if dataframe["minima"].iloc[-3] == 1 and dataframe["minima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Minima Detected - Potential Long!!!")

        return dataframe

    # ======================================================================
    # Entry / exit
    # ======================================================================

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Pull cached dynamic thresholds (computed once in populate_indicators)
        rsi_low_adj = df["rsi_low_adj"].iat[-1] if "rsi_low_adj" in df.columns else 0.0
        rsi_high_adj = df["rsi_high_adj"].iat[-1] if "rsi_high_adj" in df.columns else 0.0

        rsi_long_threshold = 30 + rsi_low_adj
        rsi_short_threshold = 70 + rsi_high_adj

        # debug-level only — runs every candle for every pair, was spamming logs at info
        logger.debug(
            f"{metadata['pair']} Entry thresholds - "
            f"Long: RSI < {rsi_long_threshold:.1f}, Short: RSI > {rsi_short_threshold:.1f}"
        )

        # ---- Filter toggles (set in config to disable individual filters) ----
        use_continuous_filter = self.config.get("filter_use_continuous", True)
        use_check_trend_filter = self.config.get("filter_use_check_trend", True)
        use_volume_filter = self.config.get("filter_use_volume", False)  # off by default
        check_upper_max = self.config.get("filter_check_upper_max", 2)
        check_down_max = self.config.get("filter_check_down_max", 2)

        # Build common filter masks — True means "candle is OK to enter on"
        long_filter = pd.Series(True, index=df.index)
        short_filter = pd.Series(True, index=df.index)

        if use_continuous_filter:
            # Don't long when price has been pumping for N straight candles (chase risk)
            long_filter &= (df["continuous_up"] == 0)
            # Don't short when price has been dumping for N straight candles
            short_filter &= (df["continuous_down"] == 0)

        if use_check_trend_filter:
            # Don't long into a sustained uptrend (overextended)
            long_filter &= (df["check_upper"] < check_upper_max)
            # Don't short into a sustained downtrend
            short_filter &= (df["check_down"] < check_down_max)

        if use_volume_filter:
            # Optional: only act when there's been a recent volume confirmation
            long_filter &= (df["volume_change_check"] == 1)
            short_filter &= (df["volume_change_check"] == 1)

        # NOTE: v19's extra short-side entry blockers (uptrend-block + BTC-bullish
        # gate) were tried in 19a and killed too many shorts. Reverted here to
        # v18 behaviour. The runaway-trend protection now lives ENTIRELY on the
        # exit side (custom_exit + strong_uptrend / strong_downtrend), so shorts
        # fire as in v18 but get pulled out fast when the move goes one-sided.

        # ============ LONG ENTRIES ============

        # Minima - Standard long entry
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["maxima_check"] == 1)
                & (df["&s-extrema"] < 0)
                & (df["minima"].shift(1) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] < rsi_long_threshold)
                & long_filter
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima")

        # Minima Full Send - Aggressive long entry
        df.loc[
            (
                (df["minima_check"] == 0)
                & (df["volume"] > 0)
                & (df["rsi"] < rsi_long_threshold)
                & long_filter
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Full Send")

        # Minima Check - Confirmation long entry
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["minima_check"] == 0)
                & (df["minima_check"].shift(5) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] < rsi_long_threshold)
                & long_filter
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Minima Check")

        # ============ SHORT ENTRIES ============
        # NOTE: can_short = False at the class level; these only fire if you flip it.

        # Maxima - Standard short entry
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["minima_check"] == 1)
                & (df["&s-extrema"] > 0)
                & (df["maxima"].shift(1) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] > rsi_short_threshold)
                & short_filter
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima")

        # Maxima Full Send - Aggressive short entry
        df.loc[
            (
                (df["maxima_check"] == 0)
                & (df["volume"] > 0)
                & (df["rsi"] > rsi_short_threshold)
                & short_filter
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Full Send")

        # Maxima Check - Confirmation short entry
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["maxima_check"] == 0)
                & (df["maxima_check"].shift(5) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] > rsi_short_threshold)
                & short_filter
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Maxima Check")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Pull cached dynamic thresholds
        rsi_low_adj = df["rsi_low_adj"].iat[-1] if "rsi_low_adj" in df.columns else 0.0
        rsi_high_adj = df["rsi_high_adj"].iat[-1] if "rsi_high_adj" in df.columns else 0.0

        # Exit long when RSI is HIGH (overbought)
        rsi_exit_long_threshold = 70 + rsi_high_adj
        # Exit short when RSI is LOW (oversold)
        rsi_exit_short_threshold = 30 + rsi_low_adj

        logger.debug(
            f"{metadata['pair']} Exit thresholds - "
            f"Long: RSI > {rsi_exit_long_threshold:.1f}, Short: RSI < {rsi_exit_short_threshold:.1f}"
        )

        # ============ EXIT LONG ============

        # Maxima detected + overbought
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["&s-extrema"] > 0)
                & (df["maxima"].shift(1) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] > rsi_exit_long_threshold)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Maxima")

        # Maxima Full Send + overbought
        df.loc[
            (
                (df["maxima_check"] == 0)
                & (df["volume"] > 0)
                & (df["rsi"] > rsi_exit_long_threshold)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Maxima Full Send")

        # ============ EXIT SHORT ============

        # Minima detected + oversold
        df.loc[
            (
                (df["DI_catch"] == 1)
                & (df["&s-extrema"] < 0)
                & (df["minima"].shift(1) == 1)
                & (df["volume"] > 0)
                & (df["rsi"] < rsi_exit_short_threshold)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "Minima")

        # Minima Full Send + oversold
        df.loc[
            (
                (df["minima_check"] == 0)
                & (df["volume"] > 0)
                & (df["rsi"] < rsi_exit_short_threshold)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "Minima Full Send")
        # NOTE: v18's "Reversal: Long signal" / "Reversal: Short signal" block
        # was moved out of here in v19. With exit_profit_only=True freqtrade
        # silently drops exit_long/exit_short signals when the trade is at a
        # loss, so reversals on deep-loss positions (the TON case) never fired.
        # See `custom_exit` below — it bypasses exit_profit_only.
        return df


# ==========================================================================
# Helper functions (module level)
# ==========================================================================


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
