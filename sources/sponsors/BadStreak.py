from jesse.strategies import Strategy
from jesse import utils
import jesse.indicators as ta
import custom_indicators as cta
from datetime import datetime, timezone, timedelta


# Tested with route: ('Binance', 'LTC-USDT', '1h', 'BadStreak')
class BadStreak(Strategy):
	def __init__(self):
		super().__init__()
		self.vars['trade_allowed'] = True
		self.vars['prevent_trade_till_date'] = None

	@property
	def date(self):
		return datetime.fromtimestamp(int(self.current_candle[0]) / 1000, tz=timezone.utc)

	# Momentum Pinball
	@property
	def mp(self):
		return cta.mp(self.candles[:, 2], roc_period=1, rsi_period=3, sequential=False)

	# Moving Average Cross (Market Color)
	@property
	def mac(self):
		return cta.mac(self.candles, 20, 50, sequential=False)

	# MA Streak
	@property
	def ma_streak(self):
		return cta.ma_streak(self.candles, ma_period=4, sequential=False)

	# Percent Change Channel
	@property
	def pcc(self):
		return cta.pcc(self.candles, period=20, mult=2, sequential=False)

	# Streak ROC
	@property
	def streak_roc(self):
		streak = abs(int(self.ma_streak))
		streak_back_close = self.candles[:, 2][-(streak + 1)]

		return 100 * (self.close - streak_back_close) / streak_back_close

	@property
	def fast_ema(self):
		return ta.ema(self.candles, period=34, source_type="close")

	@property
	def slow_ema(self):
		return ta.ema(self.candles, period=55, source_type="close")

	def should_long(self) -> bool:
		return self.mp < 30

	def should_short(self) -> bool:
		return self.mp > 70

	def should_cancel(self) -> bool:
		return True

	def is_trade_allowed_filter(self):
		return self.vars['trade_allowed']

	def slow_fast_ema_filter(self):
		if self.should_long():
			return self.fast_ema > self.slow_ema
		else:
			return self.fast_ema < self.slow_ema

	def market_color_filter(self):
		if self.should_long():
			return self.mac == 1
		else:
			return self.mac == -1

	def streak_filter(self):
		if self.should_long():
			return self.streak_roc > self.pcc.lowerband
		else:
			return self.streak_roc < self.pcc.upperband

	def filters(self):
		return [
			self.is_trade_allowed_filter,
			self.streak_filter,
			self.market_color_filter
		]

	def go_long(self):
		qty = utils.size_to_qty(self.capital * 0.1, self.price, fee_rate=self.fee_rate)

		self.buy = qty, self.price

		self.take_profit = [
			(qty / 3, self.price + (self.price * 0.01)),
			(qty / 3, self.price + (self.price * 0.02)),
			(qty / 3, self.price + (self.price * 0.03))
		]

	def go_short(self):
		qty = utils.size_to_qty(self.capital * 0.1, self.price, fee_rate=self.fee_rate)

		self.sell = qty, self.price

		self.take_profit = [
			(qty / 3, self.price - (self.price * 0.01)),
			(qty / 3, self.price - (self.price * 0.02)),
			(qty / 3, self.price - (self.price * 0.03))
		]

	def update_position(self):
		if self.is_long:
			if self.close > self.average_entry_price + (self.average_entry_price * 0.02):
				self.liquidate()
		if self.is_short:
			if self.close < self.average_entry_price - (self.average_entry_price * 0.02):
				self.liquidate()

	def on_reduced_position(self, order):
		if self.is_long:
			self.stop_loss = self.position.qty, self.price - (self.price * 0.01)
		elif self.is_short:
			self.stop_loss = self.position.qty, self.price + (self.price * 0.01)

	def on_stop_loss(self, order):
		self.vars['trade_allowed'] = False
		self.vars['prevent_trade_till_date'] = self.date + timedelta(days=1)

	def before(self):
		if self.date == self.vars['prevent_trade_till_date']:
			self.vars['trade_allowed'] = True
