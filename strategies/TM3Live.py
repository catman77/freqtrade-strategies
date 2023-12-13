# add common folders to path
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# add common folders to path
import warnings
import logging
from functools import reduce
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence.trade_model import Order, Trade
from freqtrade.strategy.parameters import CategoricalParameter, DecimalParameter, IntParameter
from typing import Dict, List, Optional

from user_data.strategies.TM3Base import TM3Base

logger = logging.getLogger(__name__)

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class TM3Live(TM3Base):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """
    minimal_roi = {"360": 0}

    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    ignore_roi_if_entry_signal = True

    stoploss = -0.04
    trailing_stop = False
    trailing_only_offset_is_reached  = False
    trailing_stop_positive_offset = 0

    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    # internally freqtrade multiply it by 2, so we put here 1/2 of the max startup candle count
    startup_candle_count: int = 100

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 1,
                "trade_limit": 1,
                "stop_duration_candles": 24,
                "required_profit": -0.005,
                "only_per_pair": True,
                "only_per_side": True
            }
        ]

    LONG_ENTRY_SIGNAL_TRESHOLD = DecimalParameter(0.7, 0.95, decimals=2, default=0.8, space="buy", optimize=True)
    SHORT_ENTRY_SIGNAL_TRESHOLD = DecimalParameter(0.7, 0.95, decimals=2, default=0.8, space="buy", optimize=True)

    ENTRY_STRENGTH_TRESHOLD = DecimalParameter(0.4, 0.7, decimals=2, default=0.3, space="buy", optimize=True)

    LONG_TP = DecimalParameter(0.01, 0.03, decimals=3, default=0.016, space="sell", optimize=True)
    SHORT_TP = DecimalParameter(0.01, 0.03, decimals=3, default=0.016, space="sell", optimize=True)

    def protection_di(self, df: DataFrame):
        return (df["DI_values"] < df["DI_cutoff"])

    def signal_entry_long(self, df: DataFrame):
        return (df["&-trend"] >= 0.7) & (df["&s-extrema"] <= -0.3) & (df["&-s_max"] > 0.02)

    def signal_entry_short(self, df: DataFrame):
        return (df["&-trend"] <= -0.7) & (df["&s-extrema"] >= 0.3) & (df["&-s_min"] < -0.02)



    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                self.signal_entry_long(df)
            ),
            'enter_long'] = 1

        df.loc[
            (
                self.signal_entry_short(df)
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (self.signal_entry_short(df) | (df["&-s_min"] < -0.01)),
            'exit_long'] = 1

        df.loc[
            (self.signal_entry_long(df) | (df["&-s_max"] > 0.01)),
            'exit_short'] = 1

        return df


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        is_short = trade.is_short == True
        is_long = trade.is_short == False
        is_profitable = current_profit > 0
        is_short_signal = last_candle["&-trend"] <= -1
        is_long_signal = last_candle["&-trend"] >= 1

        # exit on profit target & if not entry signal
        if trade.is_open and is_long and (current_profit >= self.LONG_TP.value) and not is_long_signal:
            return "long_profit_target_reached"

        if trade.is_open and is_short and (current_profit >= self.SHORT_TP.value) and not is_short_signal:
            return "short_profit_target_reached"

