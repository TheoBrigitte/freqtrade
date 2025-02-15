from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, informative
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timezone
from freqtrade.persistence import Trade
from pandas import Series
import logging
from functools import reduce

logger = logging.getLogger(__name__)

class FVGAdvancedStrategy_V2(IStrategy):
    """
    高级FVG策略 - 支持多空双向交易、DCA、动态杠杆和趋势跟踪
    """
    INTERFACE_VERSION = 3
    
    # 启用做空
    can_short = False

    # 基础策略参数
    minimal_roi = {
        "0": 0.3
    }
    stoploss = -0.3
    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 杠杆参数
    max_leverage = 5.0
    min_leverage = 1.0
    default_leverage = 2.0
    
    # DCA参数
    max_dca_orders = 2
    max_dca_multiplier = 2.0
    minimal_dca_profit = -0.05
    max_dca_profit = -0.1

    # 分批减仓参数
    position_adjustment_enable = True
    max_entry_position_adjustment = -1
    max_exit_position_adjustment = 3
    exit_portion_size = 0.3

    # 指标参数
    filter_width = DecimalParameter(0.0, 5.0, default=0.3, space='buy')
    tp_mult = DecimalParameter(1.0, 10.0, default=4.0, space='sell')
    sl_mult = DecimalParameter(1.0, 5.0, default=2.0, space='sell')

    # 市场状态评估参数
    rsi_period = 14
    bb_period = 20
    bb_std = 2
    adx_period = 14
    volatility_period = 100

    # 趋势判断参数
    trend_ema_period = 20
    trend_rsi_period = 14
    trend_adx_period = 14
    trend_adx_threshold = 30

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("FVG Advanced Strategy Initialized")

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        1小时趋势判断指标
        """
        # EMA趋势
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.trend_ema_period)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.trend_rsi_period)
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.trend_adx_period)
        
        # 趋势判断
        dataframe['uptrend'] = (
            (dataframe['close'] > dataframe['ema']) &  # 价格在EMA上方
            (dataframe['rsi'] > 50) &                  # RSI大于50
            (dataframe['adx'] > self.trend_adx_threshold)  # ADX大于阈值
        )
        
        dataframe['downtrend'] = (
            (dataframe['close'] < dataframe['ema']) &  # 价格在EMA下方
            (dataframe['rsi'] < 50) &                  # RSI小于50
            (dataframe['adx'] > self.trend_adx_threshold)  # ADX大于阈值
        )
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR计算
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=200)
        
        # FVG计算所需的移位数据
        dataframe['low_3'] = dataframe['low'].shift(3)
        dataframe['high_1'] = dataframe['high'].shift(1)
        dataframe['close_2'] = dataframe['close'].shift(2)
        dataframe['high_3'] = dataframe['high'].shift(3)
        dataframe['low_1'] = dataframe['low'].shift(1)

        # 计算bullish FVG
        dataframe['bull_fvg'] = (
            (dataframe['low_3'] > dataframe['high_1']) &
            (dataframe['close_2'] < dataframe['low_3']) &
            (dataframe['close'] > dataframe['low_3']) &
            ((dataframe['low_3'] - dataframe['high_1']) > dataframe['atr'] * self.filter_width.value)
        )

        # 计算bearish FVG
        dataframe['bear_fvg'] = (
            (dataframe['low_1'] > dataframe['high_3']) &
            (dataframe['close_2'] > dataframe['high_3']) &
            (dataframe['close'] < dataframe['high_3']) &
            ((dataframe['low_1'] - dataframe['high_3']) > dataframe['atr'] * self.filter_width.value)
        )

        # 计算FVG平均值
        dataframe['bull_avg'] = (dataframe['low_3'] + dataframe['high_1']) / 2
        dataframe['bear_avg'] = (dataframe['low_1'] + dataframe['high_3']) / 2

        # 市场状态指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        
        bollinger = ta.BBANDS(dataframe)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period)
        dataframe['volatility'] = dataframe['close'].rolling(window=self.volatility_period).std()
        
        # 计算市场状态得分
        dataframe['market_score'] = self.calculate_market_score(dataframe)

        return dataframe

    def calculate_market_score(self, dataframe: DataFrame) -> Series:
        """
        计算市场状态得分
        """
        score = pd.Series(index=dataframe.index, data=0.0)

        # ADX评分 (0-25)
        adx_score = dataframe['adx'] / 100 * 25
        score += adx_score

        # 波动性评分 (0-25)
        vol_normalized = (dataframe['volatility'] - dataframe['volatility'].min()) / \
                        (dataframe['volatility'].max() - dataframe['volatility'].min())
        volatility_score = (1 - vol_normalized) * 25
        score += volatility_score

        # RSI评分 (0-25)
        rsi_score = (1 - abs(dataframe['rsi'] - 50) / 50) * 25
        score += rsi_score

        # 布林带宽度评分 (0-25)
        bb_score = (1 - dataframe['bb_width']) * 25
        score += bb_score

        return score

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # 获取1小时数据的趋势信息
        informative = self.dp.get_pair_dataframe(metadata['pair'], '1h')

        # 多头入场信号
        dataframe.loc[
            (
                dataframe['bull_fvg']
                & dataframe['uptrend_1h']
                & (dataframe['market_score'] > 60)
            ),
            'enter_long'] = 1

        # 空头入场信号
        dataframe.loc[
            (
                dataframe['bear_fvg']
                & dataframe['downtrend_1h']
                & (dataframe['market_score'] < 40)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # 获取1小时数据的趋势信息
        informative = self.dp.get_pair_dataframe(metadata['pair'], '1h')

        # 多头退出信号
        dataframe.loc[
            (
                # (dataframe['close'] > dataframe['bull_avg'] + dataframe['atr'] * self.tp_mult.value) |
                # (dataframe['close'] < dataframe['bull_avg'] - dataframe['atr'] * self.sl_mult.value) |
                # dataframe['downtrend_1h'] |
                dataframe['bear_fvg']
            ),
            'exit_long'] = 1

        # 空头退出信号
        dataframe.loc[
            (
                # (dataframe['close'] < dataframe['bear_avg'] - dataframe['atr'] * self.tp_mult.value) |
                # (dataframe['close'] > dataframe['bear_avg'] + dataframe['atr'] * self.sl_mult.value) |
                # dataframe['uptrend_1h'] |
                dataframe['bull_fvg']
            ),
            'exit_short'] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """
        处理DCA和分批减仓
        """
        if current_profit <= self.minimal_dca_profit and current_profit >= self.max_dca_profit:
            filled_entries = trade.select_filled_orders('enter_short' if trade.is_short else 'enter_long')
            count_entries = len(filled_entries)
            
            if count_entries < self.max_dca_orders:
                stake_amount = trade.stake_amount * self.max_dca_multiplier
                return stake_amount

        elif current_profit > 0.05:
            filled_exits = trade.select_filled_orders('exit_short' if trade.is_short else 'exit_long')
            count_exits = len(filled_exits)
            
            if count_exits < self.max_exit_position_adjustment:
                return -(trade.amount * self.exit_portion_size)

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        动态杠杆管理
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        informative = self.dp.get_pair_dataframe(pair, '1h')
        
        if len(dataframe) == 0 or len(informative) == 0:
            return self.default_leverage

        current_market_score = dataframe['market_score'].iloc[-1]
        current_rsi = dataframe['rsi'].iloc[-1]
        trend_adx = dataframe['adx_1h'].iloc[-1]
        
        # 基于市场得分和趋势强度调整杠杆
        if current_market_score >= 80 and trend_adx > 30:
            leverage = self.max_leverage
        elif current_market_score >= 60 and trend_adx > 25:
            leverage = self.max_leverage * 0.8
        elif current_market_score >= 40:
            leverage = self.default_leverage
        elif current_market_score >= 20:
            leverage = self.min_leverage * 1.5
        else:
            leverage = self.min_leverage

        # 根据RSI和交易方向调整杠杆
        if side == "long":
            if current_rsi < 30 and dataframe['uptrend_1h'].iloc[-1]:
                leverage *= 1.2
            elif current_rsi > 70 or dataframe['downtrend_1h'].iloc[-1]:
                leverage *= 0.8
        else:  # side == "short"
            if current_rsi > 70 and dataframe['downtrend_1h'].iloc[-1]:
                leverage *= 1.2
            elif current_rsi < 30 or dataframe['uptrend_1h'].iloc[-1]:
                leverage *= 0.8

        return round(min(max(leverage, self.min_leverage), self.max_leverage), 1)

    # def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
    #                current_profit: float, **kwargs) -> Optional[str]:
    #     """
    #     市场状况恶化或趋势反转时的提前退出机制
    #     """
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     # informative = self.dp.get_pair_dataframe(pair, '1h')
        
    #     if len(dataframe) == 0:
    #         return None
            
    #     current_market_score = dataframe['market_score'].iloc[-1]
        
    #     # 趋势反转退出
    #     if not trade.is_short and dataframe['downtrend_1h'].iloc[-1]:
    #         return "1h_trend_reversal_short"
    #     elif trade.is_short and dataframe['uptrend_1h'].iloc[-1]:
    #         return "1h_trend_reversal_long"
        
    #     # 市场状况恶化时退出
    #     if current_market_score < 20 and current_profit > 0:
    #         return "market_condition_bad"
            
    #     # 趋势减弱时退出
    #     if dataframe['adx_1h'].iloc[-1] < 20 and current_profit > 0.02:
    #         return "trend_weakening"
            
    #     # 波动性突增时退出
    #     if dataframe['bb_width'].iloc[-1] > dataframe['bb_width'].iloc[-2] * 1.5:
    #         return "volatility_increasing"
            
    #     return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: float, max_stake: float,
                          entry_tag: str, **kwargs) -> float:
        """
        自定义每次交易的资金量
        """
        return proposed_stake
