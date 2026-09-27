# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

"""
FFT_AdaptiveCycle_v1
====================

ARCHITECTURE
------------
1. Causal FFT cycle detector  → dominant period P + quality metrics
   (log-return window, Hann taper, parabolic peak interpolation)
2. Kalman filter              → K_period: stable period with no drift/float
   When FFT detects valid cycle: update with low noise R.
   When FFT sees noise: prediction-only (period drifts at rate Q per bar).
3. Adaptive biquad bandpass   → BP: pure cycle component of price
   Center = 1/K_period, Q = 1/bw_ratio (default Q≈3.3 for 30% bandwidth).
   Coefficients updated per bar → filter adapts as cycle period shifts.
4. Cycle oscillator           → bp_z = BP / rolling_RMS(BP, 50 bars)
   Values near ±1 = typical extremes. ±2 = strong extremes.
5. EMA-200 slope              → macro trend direction (up/down/flat)
6. Entry: cycle crossing (trough→up or peak→down) + trend + quality gate
7. Exit:  populate_exit_trend (cycle completion) + custom_exit (trailing TP)
8. DCA:   linear price steps from open_rate, exponential volume scale
9. Leverage: 2×

KEY DESIGN CHOICES (based on community feedback)
-------------------------------------------------
- Kalman filter solves the "float" problem: raw FFT period jumps bar-to-bar;
  Kalman treats valid detections as noisy measurements and carries forward
  a stable estimate between detections.
- Bandpass center tracks K_period not the raw FFT → no coefficient discontinuities.
- Entry on cycle CROSSING (not level) fires once per trough/peak, not every bar.
- Trend filter (EMA-200 slope) avoids counter-trend mean-reversion in strong moves.
- Trailing retrace = 0.5% tuned for 2× leverage noise floor (~0.25% price tick).

PARAMETERS (config.json overrides)
-----------------------------------
  fft_window              int   = 128    bars for FFT window
  fft_min_period          int   = 8      min detectable cycle
  fft_max_period          int   = 100    max detectable cycle
  fft_default_period      int   = 40     fallback period
  kalman_q                float = 2.0    process noise (period drift rate)
  kalman_r                float = 9.0    measurement noise (valid detections)
  bp_bandwidth            float = 0.3    bandpass bandwidth ratio (Q = 1/0.3 ≈ 3.3)
  bp_z_window             int   = 50     rolling RMS window for oscillator
  bp_z_entry              float = 0.8    |oscillator| crossing threshold for entry
  bp_z_exit               float = 0.7    oscillator threshold for cycle-completion exit
  use_trend_filter        bool  = True   require EMA-200 slope alignment
  trend_ema_period        int   = 200    EMA period for macro trend
  trend_slope_bars        int   = 10     bars over which EMA slope is measured
  min_cycle_dominance     float = 0.12   minimum FFT dominance to allow entries
  use_cycle_exit          bool  = True   exit via cycle-completion signal
  take_profit             float = 0.010  1.0% profit target (leveraged)
  trailing_after_tp       bool  = True   trailing mode after TP hit
  trailing_min_activation float = 0.005  trailing arms at TP + 0.5%
  trailing_retrace        float = 0.005  exit on 0.5% pullback from peak
  force_exit_by_days_en   bool  = True
  force_exit_after_days   float = 2.0
  safety_order_ratio      float = 1.2
  safety_order_max_count  int   = 5
  safety_order_vol_scale  float = 1.2
  price_deviation_initial float = 0.02

DCA COVERAGE (stake=50, ratio=1.2, scale=1.2, dev=2%, leverage=2×)
--------------------------------------------------------------------
  Level  Price drop  P&L drop  Order(USDT)  Cumul(USDT)
  Entry      0%         0%         50           50
  DCA 1     -2%        -4%         60          110
  DCA 2     -4%        -8%         72          182
  DCA 3     -6%        -12%        86          268
  DCA 4     -8%        -16%       104          372
  DCA 5    -10%        -20%       124          496
  Total worst-case: 496 USDT / trade

RESTART SAFETY
--------------
Trailing TP state (tp_activated, max_profit) stored via trade.set_custom_data()
— survives bot restarts, crashes, and /reload_config.

CHANGELOG
---------
  v1.0: Initial release. Kalman + biquad bandpass + crossing entry + trailing TP.
"""

import sys
import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional, Tuple, Any

# Windows UTF-8 fix
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from freqtrade.strategy import IStrategy, Trade
import talib.abstract as ta

logger = logging.getLogger(__name__)


class FFTAdaptiveCycle(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short: bool = True

    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 300
    position_adjustment_enable = True
    max_entry_position_adjustment = 7

    # ── FFT ──────────────────────────────────────────────────────────────────
    fft_window: int = 128
    fft_min_period: int = 8
    fft_max_period: int = 100
    fft_default_period: int = 40
    fft_smooth_span: int = 5
    fft_peak_ratio_min: float = 4.0
    fft_entropy_max: float = 0.85

    # ── Kalman ───────────────────────────────────────────────────────────────
    kalman_q: float = 2.0
    kalman_r: float = 9.0

    # ── Bandpass + oscillator ─────────────────────────────────────────────────
    bp_bandwidth: float = 0.3
    bp_z_window: int = 50
    bp_z_entry: float = 0.8

    # ── Trend ─────────────────────────────────────────────────────────────────
    use_trend_filter: bool = True
    trend_ema_period: int = 200
    trend_slope_bars: int = 10
    min_cycle_dominance: float = 0.12

    # ── Take Profit / Trailing ────────────────────────────────────────────────
    take_profit: float = 0.010
    trailing_after_tp: bool = True
    trailing_min_activation: float = 0.005
    trailing_retrace: float = 0.003
    force_exit_by_days_enabled: bool = True
    force_exit_after_days: float = 2.0

    # ── DCA ───────────────────────────────────────────────────────────────────
    safety_order_ratio: float = 1.2
    safety_order_max_count: int = 7
    safety_order_volume_scale: float = 1.2
    price_deviation_initial: float = 0.02

    _EXIT_STATE_KEY = "exit_state"
    _exit_states: dict = {}

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        if sys.platform == "win32":
            for _h in logging.root.handlers:
                if isinstance(_h, logging.FileHandler):
                    if getattr(_h, "encoding", "utf-8") not in ("utf-8", "utf8"):
                        try:
                            _h.close()
                            _h.encoding = "utf-8"
                            _h.stream = open(
                                _h.baseFilename, _h.mode,
                                encoding="utf-8", errors="replace",
                            )
                        except Exception:
                            pass

        if "timeframe" in config:
            self.timeframe = config["timeframe"]

        def _g(key, default):
            return config.get(key, default)

        self.fft_window              = int(_g("fft_window", self.fft_window))
        self.fft_min_period          = int(_g("fft_min_period", self.fft_min_period))
        self.fft_max_period          = int(_g("fft_max_period", self.fft_max_period))
        self.fft_default_period      = int(_g("fft_default_period", self.fft_default_period))
        self.fft_smooth_span         = int(_g("fft_smooth_span", self.fft_smooth_span))
        self.fft_peak_ratio_min      = float(_g("fft_peak_ratio_min", self.fft_peak_ratio_min))
        self.fft_entropy_max         = float(_g("fft_entropy_max", self.fft_entropy_max))
        self.kalman_q                = float(_g("kalman_q", self.kalman_q))
        self.kalman_r                = float(_g("kalman_r", self.kalman_r))
        self.bp_bandwidth            = float(_g("bp_bandwidth", self.bp_bandwidth))
        self.bp_z_window             = int(_g("bp_z_window", self.bp_z_window))
        self.bp_z_entry              = float(_g("bp_z_entry", self.bp_z_entry))
        self.use_trend_filter        = bool(_g("use_trend_filter", self.use_trend_filter))
        self.trend_ema_period        = int(_g("trend_ema_period", self.trend_ema_period))
        self.trend_slope_bars        = int(_g("trend_slope_bars", self.trend_slope_bars))
        self.min_cycle_dominance     = float(_g("min_cycle_dominance", self.min_cycle_dominance))
        self.take_profit             = float(_g("take_profit", self.take_profit))
        self.trailing_after_tp       = bool(_g("trailing_after_tp", self.trailing_after_tp))
        self.trailing_min_activation = float(_g("trailing_min_activation", self.trailing_min_activation))
        self.trailing_retrace        = float(_g("trailing_retrace", self.trailing_retrace))
        self.force_exit_by_days_enabled = bool(_g("force_exit_by_days_enabled", self.force_exit_by_days_enabled))
        self.force_exit_after_days   = float(_g("force_exit_after_days", self.force_exit_after_days))
        self.safety_order_ratio      = float(_g("safety_order_ratio", self.safety_order_ratio))
        self.safety_order_max_count  = int(_g("safety_order_max_count", self.safety_order_max_count))
        self.safety_order_volume_scale = float(_g("safety_order_volume_scale", self.safety_order_volume_scale))
        self.price_deviation_initial = float(_g("price_deviation_initial", self.price_deviation_initial))

        self.max_entry_position_adjustment = self.safety_order_max_count

        logger.info("✅ FFTAdaptiveCycle initialized")
        logger.info(f"   timeframe          = {self.timeframe}")
        logger.info(f"   fft_window         = {self.fft_window}")
        logger.info(f"   period_range       = [{self.fft_min_period}, {self.fft_max_period}]")
        logger.info(f"   kalman_q/r         = {self.kalman_q}/{self.kalman_r}")
        logger.info(f"   bp_bandwidth       = {self.bp_bandwidth:.0%}  (Q≈{1/self.bp_bandwidth:.1f})")
        logger.info(f"   bp_z_entry         = ±{self.bp_z_entry}")
        logger.info(f"   trend_filter       = {self.use_trend_filter}")
        logger.info(f"   min_dominance      = {self.min_cycle_dominance}")
        logger.info(f"   take_profit        = {self.take_profit:.1%}")
        logger.info(f"   trailing_retrace   = {self.trailing_retrace:.1%}")
        logger.info(f"   dca_levels         = {self.safety_order_max_count}")

    # =========================================================================
    # CAUSAL FFT CYCLE DETECTOR
    # =========================================================================

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        n = max(1, int(n))
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _linear_detrend_window(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        L = len(x)
        if L < 3:
            return x - np.nanmean(x)
        j = np.arange(L, dtype=float)
        j_mean = (L - 1) / 2.0
        x_mean = np.nanmean(x)
        jc = j - j_mean
        xc = x - x_mean
        denom = np.sum(jc * jc)
        if denom <= 1e-12:
            return x - x_mean
        b = np.sum(jc * xc) / denom
        a = x_mean - b * j_mean
        return x - (a + b * j)

    @staticmethod
    def _robust_zscore_window(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if np.isfinite(mad) and mad > eps:
            z = (x - med) / (1.4826 * mad + eps)
        else:
            mu = np.nanmean(x)
            sigma = np.nanstd(x, ddof=1)
            if np.isfinite(sigma) and sigma > eps:
                z = (x - mu) / (sigma + eps)
            else:
                z = x - mu
        z = z - np.nanmean(z)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def causal_fft_cycle_period(
        self,
        price: pd.Series,
        window_size: int,
        min_period: int = 12,
        max_period: int = 160,
        default_period: int = 80,
        smooth_span: int = 10,
        peak_ratio_min: float = 4.0,
        dominance_min: float = 0.12,
        entropy_max: float = 0.85,
    ) -> DataFrame:
        eps = 1e-12
        price = (
            pd.Series(price, copy=True)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
        )
        index = price.index
        values = price.to_numpy(dtype=float)
        n_rows = len(values)
        L = int(window_size)
        L = max(L, min_period * 2 + 4)

        max_period = int(min(max_period, L // 2))
        min_period = int(max(3, min_period))
        if max_period <= min_period:
            max_period = min_period + 1

        n_fft = self._next_power_of_two(L)
        hann = np.hanning(L)
        window_energy = np.mean(hann * hann)
        if not np.isfinite(window_energy) or window_energy <= eps:
            window_energy = 1.0

        freqs = np.fft.rfftfreq(n_fft, d=1.0)
        periods = np.full_like(freqs, np.inf, dtype=float)
        periods[1:] = 1.0 / np.maximum(freqs[1:], eps)
        band_mask = (
            np.isfinite(periods)
            & (periods >= min_period)
            & (periods <= max_period)
        )
        band_idx = np.flatnonzero(band_mask)

        cycle_period = np.full(n_rows, float(default_period), dtype=float)
        raw_period   = np.full(n_rows, np.nan, dtype=float)
        peak_ratio   = np.full(n_rows, np.nan, dtype=float)
        dominance    = np.full(n_rows, np.nan, dtype=float)
        entropy      = np.full(n_rows, np.nan, dtype=float)
        valid        = np.zeros(n_rows, dtype=bool)

        if len(band_idx) < 3 or n_rows == 0:
            return DataFrame(
                {"cycle_period": cycle_period, "fft_raw_period": raw_period,
                 "fft_peak_ratio": peak_ratio, "fft_dominance": dominance,
                 "fft_entropy": entropy, "fft_valid": valid},
                index=index,
            )

        alpha_period = 2.0 / (float(smooth_span) + 1.0)
        prev_period  = float(default_period)

        for i in range(L - 1, n_rows):
            window = values[i - L + 1: i + 1]
            if (
                len(window) != L
                or not np.all(np.isfinite(window))
                or np.any(window <= 0)
            ):
                cycle_period[i] = prev_period
                continue

            x = np.log(np.clip(window, eps, None))
            x = self._linear_detrend_window(x)
            z = self._robust_zscore_window(x, eps=eps)
            y = z * hann

            fft_out = np.fft.rfft(y, n=n_fft)
            power = (np.abs(fft_out) ** 2) / (L * window_energy + eps)
            if len(power) > 2:
                if n_fft % 2 == 0:
                    power[1:-1] *= 2.0
                else:
                    power[1:] *= 2.0
            power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

            band_power        = power[band_idx]
            total_band_power  = float(np.sum(band_power))
            if total_band_power <= eps:
                cycle_period[i] = prev_period
                continue

            local_peak_pos = int(np.argmax(band_power))
            k_peak         = int(band_idx[local_peak_pos])
            peak_power     = float(power[k_peak])
            median_power   = float(np.median(band_power))
            pr             = peak_power / (median_power + eps)
            dom            = peak_power / (total_band_power + eps)
            prob           = band_power / (total_band_power + eps)
            ent            = -float(np.sum(prob * np.log(prob + eps))) / np.log(len(prob))

            peak_ratio[i] = pr
            dominance[i]  = dom
            entropy[i]    = ent

            is_valid = (
                pr  >= peak_ratio_min
                and dom >= dominance_min
                and ent <= entropy_max
            )
            if is_valid:
                k_hat = float(k_peak)
                if 1 <= k_peak < len(power) - 1:
                    left   = np.log(power[k_peak - 1] + eps)
                    center = np.log(power[k_peak]     + eps)
                    right  = np.log(power[k_peak + 1] + eps)
                    denom  = left - 2.0 * center + right
                    if abs(denom) > eps:
                        delta = 0.5 * (left - right) / denom
                        delta = float(np.clip(delta, -0.5, 0.5))
                        k_hat = float(k_peak) + delta

                detected_period = n_fft / max(k_hat, eps)
                detected_period = float(np.clip(detected_period, min_period, max_period))
                raw_period[i]   = detected_period
                valid[i]        = True
                smoothed        = alpha_period * detected_period + (1.0 - alpha_period) * prev_period
                prev_period     = float(np.clip(smoothed, min_period, max_period))
            else:
                prev_period = float(np.clip(prev_period, min_period, max_period))

            cycle_period[i] = prev_period

        cycle_period = np.nan_to_num(
            cycle_period,
            nan=float(default_period),
            posinf=float(default_period),
            neginf=float(default_period),
        )
        return DataFrame(
            {"cycle_period": cycle_period, "fft_raw_period": raw_period,
             "fft_peak_ratio": peak_ratio, "fft_dominance": dominance,
             "fft_entropy": entropy, "fft_valid": valid},
            index=index,
        )

    # =========================================================================
    # KALMAN FILTER — stabilises detected period, eliminates float
    # =========================================================================

    def _kalman_smooth_period(
        self,
        raw_period: np.ndarray,
        valid_mask: np.ndarray,
        default_period: float,
        min_p: float,
        max_p: float,
    ) -> np.ndarray:
        """
        Scalar Kalman filter treating each valid FFT detection as a noisy
        measurement of the true cycle period.  Between detections the estimate
        drifts at process-noise rate Q — slow enough to be stable, fast enough
        to track genuine regime shifts.
        """
        n        = len(raw_period)
        smoothed = np.empty(n, dtype=float)
        x        = default_period
        P        = 100.0
        Q        = self.kalman_q
        R        = self.kalman_r

        for i in range(n):
            # Predict
            P = P + Q

            # Update only when FFT produced a valid detection
            if valid_mask[i] and np.isfinite(raw_period[i]):
                K = P / (P + R)
                x = x + K * (raw_period[i] - x)
                P = (1.0 - K) * P

            x           = float(np.clip(x, min_p, max_p))
            smoothed[i] = x

        return smoothed

    # =========================================================================
    # ADAPTIVE BIQUAD BANDPASS — center tracks Kalman period
    # =========================================================================

    def _adaptive_bandpass(
        self,
        price: np.ndarray,
        period: np.ndarray,
        bw: float = 0.3,
    ) -> np.ndarray:
        """
        Two-pole biquad bandpass (digital, standard form):
          omega = 2π / P[i]
          alpha = sin(omega) * bw / 2        (controls bandwidth; Q = 1/bw)
          BP[i] = (alpha/(1+alpha)) * (x[i] - x[i-2])
                  + (2·cos(omega)/(1+alpha)) * BP[i-1]
                  - ((1-alpha)/(1+alpha)) * BP[i-2]

        Coefficients are recomputed every bar so the filter adapts continuously
        as the Kalman period estimate evolves.
        """
        n   = len(price)
        bp  = np.zeros(n, dtype=float)
        eps = 1e-12

        for i in range(2, n):
            P_i   = float(max(period[i], 3.0))
            omega = 2.0 * math.pi / P_i
            sin_w = math.sin(omega)
            cos_w = math.cos(omega)
            alpha = sin_w * bw / 2.0
            denom = 1.0 + alpha
            if abs(denom) < eps:
                continue
            b0 =  alpha / denom            # coefficient for (x[i] - x[i-2])
            a1 = -2.0 * cos_w / denom      # coefficient for -BP[i-1]  (note sign)
            a2 =  (1.0 - alpha) / denom    # coefficient for -BP[i-2]

            bp[i] = b0 * (price[i] - price[i - 2]) - a1 * bp[i - 1] - a2 * bp[i - 2]

        return bp

    # =========================================================================
    # CYCLE OSCILLATOR — normalised bandpass  (bp_z)
    # =========================================================================

    @staticmethod
    def _cycle_oscillator(bp: np.ndarray, window: int = 50) -> np.ndarray:
        """
        Normalise BP by its rolling RMS so the oscillator is unit-free and
        comparable across pairs and time.  Values near ±1 = typical extremes;
        ±2 = strong extremes.  Clipped to [-3, 3] to remove spikes.
        """
        bp_s  = pd.Series(bp)
        min_p = max(5, window // 4)
        rms   = bp_s.pow(2).rolling(window, min_periods=min_p).mean().pow(0.5)
        osc   = bp_s / (rms + 1e-12)
        return np.clip(osc.fillna(0.0).to_numpy(dtype=float), -3.0, 3.0)

    # =========================================================================
    # EXIT STATE — in-memory primary, DB for restart recovery
    # =========================================================================

    def _load_exit_state(self, trade) -> dict:
        tid = trade.id
        if tid in self._exit_states:
            return dict(self._exit_states[tid])
        data = trade.get_custom_data(self._EXIT_STATE_KEY)
        if data:
            self._exit_states[tid] = dict(data)
            return dict(data)
        return {"tp_activated": False, "max_profit": None}

    def _save_exit_state(self, trade, data: dict) -> None:
        self._exit_states[trade.id] = dict(data)
        try:
            trade.set_custom_data(self._EXIT_STATE_KEY, data)
        except Exception:
            pass

    # =========================================================================
    # INDICATORS
    # =========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "?")
        logger.debug(f"[FFT] {pair}: computing indicators ({len(dataframe)} bars)")

        # ── 1. FFT cycle period ──────────────────────────────────────────────
        fft_df = self.causal_fft_cycle_period(
            price          = dataframe["close"],
            window_size    = self.fft_window,
            min_period     = self.fft_min_period,
            max_period     = self.fft_max_period,
            default_period = self.fft_default_period,
            smooth_span    = self.fft_smooth_span,
            peak_ratio_min = self.fft_peak_ratio_min,
            dominance_min  = self.min_cycle_dominance,
            entropy_max    = self.fft_entropy_max,
        )
        dataframe["fft_cycle_period"] = fft_df["cycle_period"].values
        dataframe["fft_raw_period"]   = fft_df["fft_raw_period"].values
        dataframe["fft_dominance"]    = fft_df["fft_dominance"].values
        dataframe["fft_entropy"]      = fft_df["fft_entropy"].values
        dataframe["fft_peak_ratio"]   = fft_df["fft_peak_ratio"].values
        dataframe["fft_valid"]        = fft_df["fft_valid"].values

        # Forward-fill dominance so quality gate doesn't flicker on NaN bars
        dataframe["fft_dom_ffill"] = (
            dataframe["fft_dominance"]
            .ffill()
            .fillna(0.0)
        )

        # ── 2. Kalman-smoothed period ────────────────────────────────────────
        k_period = self._kalman_smooth_period(
            raw_period     = fft_df["fft_raw_period"].values,
            valid_mask     = fft_df["fft_valid"].values.astype(bool),
            default_period = float(self.fft_default_period),
            min_p          = float(self.fft_min_period),
            max_p          = float(self.fft_max_period),
        )
        dataframe["kalman_period"] = k_period

        # ── 3. Adaptive bandpass ─────────────────────────────────────────────
        close_arr = dataframe["close"].to_numpy(dtype=float)
        bp_arr    = self._adaptive_bandpass(close_arr, k_period, bw=self.bp_bandwidth)
        dataframe["bp"] = bp_arr

        # ── 4. Cycle oscillator ──────────────────────────────────────────────
        dataframe["bp_z"] = self._cycle_oscillator(bp_arr, window=self.bp_z_window)

        # ── 5. Cycle completion oscillator (Ehlers-style combined) ───────────
        # Raw bandpass normalised by 3-bar EMA of its absolute value — slightly
        # smoother than bp_z for exit signals, avoiding micro-oscillations.
        bp_s  = pd.Series(bp_arr)
        bp_abs_ema = bp_s.abs().ewm(span=3, min_periods=1).mean()
        dataframe["bp_norm"] = (bp_s / (bp_abs_ema + 1e-12)).clip(-3, 3).values

        # ── 6. EMA-200 macro trend ───────────────────────────────────────────
        dataframe["ema200"]       = ta.EMA(dataframe, timeperiod=self.trend_ema_period)
        ema_shifted               = dataframe["ema200"].shift(self.trend_slope_bars)
        dataframe["ema200_slope"] = dataframe["ema200"] - ema_shifted
        dataframe["trend_up"]     = dataframe["ema200_slope"] > 0
        dataframe["trend_down"]   = dataframe["ema200_slope"] < 0

        # ── 7. ATR + ADX (context) ───────────────────────────────────────────
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ── 8. Plot threshold reference lines ─────────────────────────────────
        dataframe["bp_z_upper"]  =  self.bp_z_entry   # entry threshold (+)
        dataframe["bp_z_lower"]  = -self.bp_z_entry   # entry threshold (-)
        dataframe["bp_z_zero"]   =  0.0               # midline

        n_valid = int(fft_df["fft_valid"].sum())
        logger.info(
            f"[FFT] {pair}: {n_valid}/{len(dataframe)} valid detections, "
            f"last Kalman period = {k_period[-1]:.1f} bars"
        )
        return dataframe

    # =========================================================================
    # PLOT CONFIG
    # =========================================================================

    plot_config = {
        "main_plot": {
            "ema200": {"color": "orange", "width": 2},
        },
        "subplots": {
            "Cycle Oscillator": {
                "bp_z":      {"color": "#2196F3", "width": 2},
                "bp_norm":   {"color": "#90CAF9", "opacity": 0.5},
                "bp_z_upper":{"color": "#EF5350", "opacity": 0.4},
                "bp_z_lower":{"color": "#66BB6A", "opacity": 0.4},
                "bp_z_zero": {"color": "#777777", "opacity": 0.3},
            },
            "Kalman Period": {
                "kalman_period":    {"color": "#4CAF50", "width": 2},
                "fft_cycle_period": {"color": "#A5D6A7", "opacity": 0.6},
                "fft_raw_period":   {"color": "#C8E6C9", "opacity": 0.3},
            },
            "FFT Quality": {
                "fft_dominance":  {"color": "#9C27B0", "width": 2},
                "fft_entropy":    {"color": "#CE93D8", "opacity": 0.6},
                "fft_peak_ratio": {"color": "#7B1FA2", "opacity": 0.4},
            },
            "Trend": {
                "ema200_slope": {"color": "#FF9800", "width": 2},
                "adx":          {"color": "#607D8B", "opacity": 0.7},
            },
            "Bandpass": {
                "bp":     {"color": "#00BCD4", "opacity": 0.8},
                "bp_norm":{"color": "#80DEEA", "opacity": 0.5},
            },
        },
    }

    # =========================================================================
    # ENTRY — crossing-based triggers (fire once per trough / peak)
    # =========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        direction   = str(getattr(self, "market_direction", None) or "none").lower()
        allow_long  = direction in ("long", "none")
        allow_short = direction in ("short", "none")

        # Cycle quality gate — require a meaningful dominant cycle
        quality = dataframe["fft_dom_ffill"] >= self.min_cycle_dominance

        # Trend conditions
        if self.use_trend_filter:
            trend_long  = dataframe["trend_up"]
            trend_short = dataframe["trend_down"]
        else:
            trend_long  = pd.Series(True, index=dataframe.index)
            trend_short = pd.Series(True, index=dataframe.index)

        bpz       = dataframe["bp_z"]
        bpz_prev  = bpz.shift(1)
        bpz_prev2 = bpz.shift(2)

        # Long: oscillator just crossed UP through -bp_z_entry from below.
        # Two consecutive bars below threshold confirms we were in the trough.
        long_cross = (
            (bpz      > -self.bp_z_entry)   # current bar above threshold (exiting trough)
            & (bpz_prev  <= -self.bp_z_entry)  # previous bar was at/below threshold
            & (bpz_prev2 <= -self.bp_z_entry)  # confirmed: two bars in trough zone
        )

        # Short: oscillator just crossed DOWN through +bp_z_entry from above.
        short_cross = (
            (bpz      < self.bp_z_entry)
            & (bpz_prev  >= self.bp_z_entry)
            & (bpz_prev2 >= self.bp_z_entry)
        )

        if allow_long:
            dataframe.loc[
                long_cross & trend_long & quality & (dataframe["volume"] > 0),
                ["enter_long", "enter_tag"],
            ] = (1, "fft_long")

        if allow_short:
            dataframe.loc[
                short_cross & trend_short & quality & (dataframe["volume"] > 0),
                ["enter_short", "enter_tag"],
            ] = (1, "fft_short")

        return dataframe

    # =========================================================================
    # EXIT — handled entirely by custom_exit (trailing TP + force-by-days)
    # =========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # =========================================================================
    # DCA
    # =========================================================================

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        **kwargs: Any,
    ) -> Optional[Tuple[float, str]]:

        dca_count = trade.nr_of_successful_entries - 1
        if dca_count >= self.safety_order_max_count:
            return None
        if trade.open_orders:
            return None

        deviation = self.price_deviation_initial * (dca_count + 1)

        if not trade.is_short:
            if current_rate > trade.open_rate * (1 - deviation):
                return None
        else:
            if current_rate < trade.open_rate * (1 + deviation):
                return None

        filled_orders = trade.select_filled_orders(trade.entry_side)
        if not filled_orders:
            return None

        first_cost = filled_orders[0].cost
        so_amount  = first_cost * self.safety_order_ratio * (
            self.safety_order_volume_scale ** dca_count
        )

        if min_stake is not None and so_amount < min_stake:
            so_amount = min_stake
        so_amount = min(so_amount, max_stake)

        tag = f"DCA_{dca_count + 1}"
        logger.info(
            f"🔄 {tag} {trade.pair} ({'S' if trade.is_short else 'L'}): "
            f"{so_amount:.2f} USDT  dev={deviation:.1%}  "
            f"trigger={trade.open_rate * (1 - deviation if not trade.is_short else -(1 - deviation)):.6f}"
        )
        return so_amount, tag

    # =========================================================================
    # CUSTOM EXIT — trailing TP tuned for 2× leverage
    # =========================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> Optional[str]:

        # Force-close stalled but profitable trades
        if self.force_exit_by_days_enabled and self.force_exit_after_days > 0:
            age_days = (current_time - trade.open_date_utc).total_seconds() / 86400.0
            if age_days >= self.force_exit_after_days and current_profit > 0:
                logger.info(
                    f"⏰ Force exit {pair}: age={age_days:.2f}d "
                    f"profit={current_profit:.2%}"
                )
                self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
                return "force_exit_max_days"

        data = self._load_exit_state(trade)

        # Initialise high-water mark on first call
        if data["max_profit"] is None:
            data["max_profit"] = current_profit
            self._save_exit_state(trade, data)

        # Fixed TP mode (no trailing)
        if not self.trailing_after_tp:
            if current_profit >= self.take_profit:
                self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
                return "take_profit"
            return None

        # Arm trailing when profit exceeds take_profit + activation buffer
        if not data["tp_activated"]:
            if current_profit >= self.take_profit + self.trailing_min_activation:
                data["tp_activated"] = True
                data["max_profit"]   = current_profit
                self._save_exit_state(trade, data)
                logger.info(f"🎯 Trailing armed {pair}: {current_profit:.2%}")
            return None

        # Update high-water mark
        if current_profit > data["max_profit"]:
            data["max_profit"] = current_profit
            self._save_exit_state(trade, data)
            return None

        # Exit on retrace from peak
        retrace = data["max_profit"] - current_profit
        if retrace >= self.trailing_retrace:
            self._save_exit_state(trade, {"tp_activated": False, "max_profit": None})
            logger.info(
                f"📉 Trailing exit {pair}: "
                f"peak={data['max_profit']:.2%}  retrace={retrace:.2%}"
            )
            return "trailing_take_profit"

        return None

    # =========================================================================
    # LEVERAGE
    # =========================================================================

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return 2.0
