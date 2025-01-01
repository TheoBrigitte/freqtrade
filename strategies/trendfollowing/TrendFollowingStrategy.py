
from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from typing import Optional

class TrendFollowingStrategy(IStrategy):
    INTERFACE_VERSION: int = 3
    minimal_roi = {
        "0": 0.15,
        "30": 0.1,
        "60": 0.05
    }

    stoploss = -0.265
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False
    can_short = True
    
    timeframe = "5m"
    
    # 关键时间点：25分钟
    key_time = 25
    
    # 用于存储25分钟后的信号
    signal_after_keytime = {}  # 格式: {pair: {'signal_time': datetime, 'position': str}}

    def bot_start(self, **kwargs) -> None:
        """初始化信号存储字典"""
        self.signal_after_keytime = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OBV
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        
        # Add your trend following indicators here
        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Add your trend following buy signals here
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'enter_long'] = 1
        
        # Add your trend following sell signals here
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Add your trend following exit signals for long positions here
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'exit_long'] = 1
        
        # Add your trend following exit signals for short positions here
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'exit_short'] = 1
        
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        
        try:
            # 计算持仓时间（分钟）
            hold_time = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # 获取当前分析的数据
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            
            # 检查是否已经超过25分钟
            if hold_time >= self.key_time:
                # 检查是否是第一次到达25分钟
                if pair not in self.signal_after_keytime:
                    # 记录当前状态
                    self.signal_after_keytime[pair] = {
                        'signal_time': current_time,
                        'position': 'long' if trade.is_long else 'short'
                    }
                    return None
                
                # 检查是否有反向信号
                if trade.is_long:
                    has_short_signal = (
                        (last_candle['close'] < last_candle['trend']) and
                        (dataframe['close'].iloc[-2] >= dataframe['trend'].iloc[-2]) and
                        (last_candle['obv'] < dataframe['obv'].iloc[-2])
                    )
                    if has_short_signal:
                        del self.signal_after_keytime[pair]  # 清除信号记录
                        return "reverse_to_short"
                    
                if trade.is_short:
                    has_long_signal = (
                        (last_candle['close'] > last_candle['trend']) and
                        (dataframe['close'].iloc[-2] <= dataframe['trend'].iloc[-2]) and
                        (last_candle['obv'] > dataframe['obv'].iloc[-2])
                    )
                    if has_long_signal:
                        del self.signal_after_keytime[pair]  # 清除信号记录
                        return "reverse_to_long"
                
                # 如果没有反向信号，继续持仓
                return None
                
        except Exception as e:
            return None
        
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """使用5倍杠杆"""
        return 5.0
