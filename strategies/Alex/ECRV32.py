from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from functools import reduce
from datetime import datetime
from typing import Optional, Union
from freqtrade.persistence import Trade

from freqtrade.exchange import timeframe_to_prev_date


class ECRV32(IStrategy):
    """
    Enhanced Crash-Resistant (ECR) Strategy (V32) — PYRAMIDING EDITION

    Changes vs V31:
    - 🔥 BIGGEST CHANGE: Replaced martingale DCA (add-to-losers) with
         PYRAMIDING (add-to-winners). No more averaging down into losses.
         Trade must be in PROFIT to add size.
    - 🔥 Pyramid factor 0.5 (half previous stake), not 2.0 (doubling)
    - 🔥 Max exposure cap reduced 15× → 4× (pyramiding needs less headroom)
    - 🔥 RSI/MACD checks flipped to confirm MOMENTUM, not exhaustion
    - Everything else (entries, exits, custom_stoploss, indicators) kept
      identical to V31 so we can isolate the impact of the pyramiding change.

    Note: parameter names like `dca_trigger_pct` and `martingale_factor` are
    kept for diff minimalism. They now describe pyramiding behavior. Rename
    later for clarity.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    startup_candle_count: int = 200

    leverage_value = 10.0  # ← CONFIGURE LEVERAGE HERE

    # ROI backstop (unchanged from V31)
    minimal_roi = {
        "0": 0.25,       # 25% profit — take it
        "480": 0.10,     # After 8h, 10% is enough
        "240": 0.06,     # After 4h, 6% is enough
        "120": 0.03,     # After 2h, 3% is enough
        "60": 0.015,     # After 1h, 1.5% is enough
    }
    use_custom_exits = True

    stoploss = -0.12

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Position adjustment (now PYRAMIDING, not DCA)
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    # ════════════════════════════════════════════════════════════════
    # PYRAMIDING SETTINGS (V32) — was martingale DCA in V31
    # ════════════════════════════════════════════════════════════════
    # V31 was: factor=2.0, trigger=-0.05, rsi=32, exposure=15.0
    # V32 is:  factor=0.5, trigger=+0.02, rsi=55, exposure=4.0
    martingale_factor = 0.5             # Add HALF previous stake (not double)
    dca_trigger_pct = 0.02              # Trade must be ≥ +2% profit to pyramid
    dca_rsi_threshold = 55              # RSI must be ABOVE this (momentum confirm)
    dca_volume_multiplier = 1.2         # Volume confirmation (unchanged)
    max_total_exposure_multiplier = 4.0 # Hard cap: 4× initial stake total
    # ════════════════════════════════════════════════════════════════

    # ATR-based fixed take-profit / stop-loss
    atr_distance = 2.0
    risk_reward_ratio = 2.0

    plot_config = {
        'main_plot': {
            'bb_upperband': {
                'color': 'orange',
                'fill_to': 'bb_lowerband',
                'fill_color': 'rgba(255, 165, 0, 0.15)'
            },
            'bb_lowerband': {'color': 'orange'},
            'bb_middleband': {'color': 'gray', 'linestyle': 'dash'},
            'ema50': {'color': 'purple'},
            'ema200': {'color': 'blue', 'linewidth': 2}
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'red'},
                'hline': [
                    {'value': 70, 'color': 'red', 'linestyle': 'dash'},
                    {'value': 30, 'color': 'green', 'linestyle': 'dash'}
                ]
            },
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
                'macdhist': {'type': 'bar', 'color': 'gray'}
            },
            "ROC": {
                'roc': {'color': 'purple'}
            },
            "ATR": {
                'atr': {'color': 'teal'}
            }
        }
    }

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.profit_peaks = {}

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return -1
        candle = dataframe.iloc[-1]
        atr_multiplier = 1.5 if candle['rsi'] > 60 else 2.0
        sl_offset = atr_multiplier * candle['atr']

        if trade.is_short:
            sl_price = trade.open_rate_avg + sl_offset
            return (current_rate - sl_price) / current_rate
        else:
            sl_price = trade.open_rate_avg - sl_offset
            return (sl_price - current_rate) / current_rate

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        actual_profit = trade.calc_profit_ratio(current_rate)
        current_profit = actual_profit

        # ── PROFIT PEAK BOOKKEEPING (kept only for memory hygiene) ──
        trade_id = f"{pair}_{trade.open_date_utc.timestamp()}"
        if trade_id not in self.profit_peaks:
            self.profit_peaks[trade_id] = current_profit
        elif current_profit > self.profit_peaks[trade_id]:
            self.profit_peaks[trade_id] = current_profit

        def log_and_exit(reason):
            try:
                if trade_id in self.profit_peaks:
                    del self.profit_peaks[trade_id]
            except Exception:
                pass
            return reason

        # ── V2-STYLE FIXED ATR TP (calculated ONCE at entry) ──
        atr_roi = trade.get_custom_data(key='atr_roi', default=None)

        if atr_roi is None:
            try:
                entry_candle_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
                entry_rows = dataframe.loc[dataframe['date'] <= entry_candle_time]
                if len(entry_rows) > 0:
                    entry_candle = entry_rows.iloc[-1]
                    atr = entry_candle['atr']
                    entry_price = trade.open_rate
                    risk_distance = self.atr_distance * atr

                    if not trade.is_short:
                        atr_roi = entry_price + (risk_distance * self.risk_reward_ratio)
                    else:
                        atr_roi = entry_price - (risk_distance * self.risk_reward_ratio)

                    trade.set_custom_data(key='atr_roi', value=atr_roi)
            except Exception:
                pass

        last = dataframe.iloc[-1]
        close = last['close']

        # Hit fixed TP target
        if atr_roi is not None:
            if not trade.is_short:
                if close >= atr_roi:
                    return log_and_exit(f"ATR_TP_{current_profit*100:.1f}%")
            else:
                if close <= atr_roi:
                    return log_and_exit(f"ATR_TP_{current_profit*100:.1f}%")

        # ── REVERSAL CONFIRMATION EXITS (NOT trailing) ──
        # Only fire on a hard reversal AFTER decent profit. Two conditions
        # required so a single overshoot candle doesn't kick you out.
        if len(dataframe) > 1:
            prev = dataframe.iloc[-2]
            if not trade.is_short:
                if (prev['rsi'] > 72
                        and prev['close'] > prev['bb_middleband']
                        and prev['macd'] < prev['macdsignal']
                        and current_profit > 0.02):
                    return log_and_exit(f"Reversal_Long_{current_profit*100:.1f}%")
            else:
                if (prev['rsi'] < 28
                        and prev['close'] < prev['bb_middleband']
                        and prev['macd'] > prev['macdsignal']
                        and current_profit > 0.02):
                    return log_and_exit(f"Reversal_Short_{current_profit*100:.1f}%")

        # ── EMERGENCY CAPS ONLY (no early profit-taking) ──
        if current_profit > 0.50:
            return log_and_exit(f"Emergency_Profit_{current_profit*100:.1f}%")
        if current_profit < -0.10:
            return log_and_exit(f"Emergency_Loss_{current_profit*100:.1f}%")

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = ta.BBANDS(
            dataframe['close'],
            timeperiod=20,
            nbdevup=2.5,
            nbdevdn=2.5,
            matype=0
        )
        dataframe['bb_upperband'] = bollinger[0]
        dataframe['bb_middleband'] = bollinger[1]
        dataframe['bb_lowerband'] = bollinger[2]

        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['volume_ema'] = ta.EMA(dataframe['volume'], timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG entries — unchanged from V31
        long_conditions = [
            (dataframe['rsi'] < 40),
            (dataframe['close'] < dataframe['bb_lowerband'] * 1.02),
            (dataframe['macd'] > dataframe['macdsignal']),
            (dataframe['volume'] > dataframe['volume_ema'] * 0.8),
            (dataframe['volume'] > 0)
        ]
        long_signal = reduce(lambda x, y: x & y, long_conditions)
        dataframe.loc[long_signal, ['enter_long', 'enter_tag']] = (1, 'long_entry')

        # SHORT entries — unchanged from V31
        if self.can_short:
            short_conditions = [
                (dataframe['rsi'] > 60),
                (dataframe['close'] > dataframe['bb_upperband'] * 0.98),
                (dataframe['macd'] < dataframe['macdsignal']),
                (dataframe['volume'] > dataframe['volume_ema'] * 0.8),
                (dataframe['volume'] > 0)
            ]
            short_signal = reduce(lambda x, y: x & y, short_conditions)
            dataframe.loc[short_signal, ['enter_short', 'enter_tag']] = (1, 'short_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # Exit long when short signal fires
        dataframe.loc[dataframe['enter_short'] == 1, 'exit_long'] = 1
        dataframe.loc[dataframe['enter_short'] == 1, 'exit_tag'] = 'Reversal_Short_Signal'

        # Exit short when long signal fires
        dataframe.loc[dataframe['enter_long'] == 1, 'exit_short'] = 1
        dataframe.loc[dataframe['enter_long'] == 1, 'exit_tag'] = 'Reversal_Long_Signal'

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        V32: PYRAMIDING — add to WINNING long trades only.
        No more averaging down into losses.
        """
        # Pyramiding disabled for shorts (same as V31)
        if trade.is_short:
            return None

        # ── V32 KEY CHANGE ──────────────────────────────────
        # V31: `if current_profit > self.dca_trigger_pct: return None`
        #      (only fired when trade was in loss ≤ -5%)
        # V32: trade must be in PROFIT (≥ +2%) to pyramid
        if current_profit < self.dca_trigger_pct:
            return None
        # ────────────────────────────────────────────────────

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty or len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        # ── V32 KEY CHANGE: confirm MOMENTUM, not exhaustion ──
        # V31 wanted RSI < 32 (oversold bounce setup)
        # V32 wants RSI > 55 (uptrend momentum)
        if last_candle['rsi'] <= self.dca_rsi_threshold:
            return None

        # Volume confirmation (unchanged from V31)
        if last_candle['volume'] <= last_candle['volume_ema'] * self.dca_volume_multiplier:
            return None

        # ── V32 NEW: MACD must confirm bullish momentum ──
        if last_candle['macd'] <= last_candle['macdsignal']:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)

        if count_of_entries >= (self.max_entry_position_adjustment + 1):
            return None

        if count_of_entries > 0:
            last_entry = filled_entries[-1]
            previous_stake = last_entry.stake_amount
        else:
            previous_stake = trade.stake_amount

        # ── V32 KEY CHANGE: factor 0.5 instead of 2.0 ──
        # V31 was doubling (martingale). V32 adds half (pyramiding).
        pyramid_stake = previous_stake * self.martingale_factor

        total_stake_so_far = sum(e.stake_amount for e in filled_entries)
        projected_total = total_stake_so_far + pyramid_stake
        initial_stake = trade.stake_amount
        if projected_total > initial_stake * self.max_total_exposure_multiplier:
            return None

        if min_stake is not None and pyramid_stake < min_stake:
            pyramid_stake = min_stake
        if max_stake is not None and pyramid_stake > max_stake:
            pyramid_stake = max_stake

        return pyramid_stake

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        return proposed_stake
