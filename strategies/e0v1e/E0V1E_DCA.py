import math
import logging
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import time
from typing import Dict, List, Optional
from E0V1E import E0V1E

logger = logging.getLogger(__name__)

class E0V1E_DCA (E0V1E):

    position_adjustment_enable = True
    max_entry = 2
    first_entry_ratio = 0.65

    stoploss = -1

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        self.proposed_stake = proposed_stake
        return proposed_stake * self.first_entry_ratio

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:
        if current_profit > -0.05:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)
        if count_of_entries >= self.max_entry: return None

        dca_amount = self.proposed_stake * (1 - self.first_entry_ratio)
        logger.info(f"DCA {trade.pair} with stake amount of: {dca_amount}")
        return dca_amount
