# add common folders to path
import sys
import os

from freqtrade.strategy.strategy_helper import merge_informative_pair

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import sdnotify
from freqtrade.enums.runmode import RunMode
from typing import Dict, List, Optional
from lib.ma import MovingAveragesCalculate, MovingAveragesCalculator2
from lib.mom import MomentumANDVolatilityCalculate
from lib.cycle import CycleCalculate
from lib.trend import TrendCalculate
from lib.oscillators import OscillatorsCalculate
from lib import helpers
from lib.sagemaster import SageMasterClient
from lib.Alpha101 import get_alpha
from scipy.special import softmax

import lib.glassnode as gn

import warnings
import json
import logging
from functools import reduce
import time
import numpy as np
from technical.pivots_points import pivots_points

import pandas as pd
from pandas import DataFrame, Series
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import BooleanParameter, DecimalParameter, IntParameter
from datetime import timedelta, datetime, timezone
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
import talib.abstract as ta
from sqlalchemy import column, desc
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import argrelextrema
from joblib import Parallel, delayed

from technical import qtpylib
import pandas_ta as pta

from lib.prediction_storage import PredictionStorage

logger = logging.getLogger(__name__)

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def candle_stats(dataframe):
    # print("candle_stats", dataframe)
    # log data
    dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
    dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] +
                          dataframe['low'] + dataframe['close']) / 4

    dataframe['hlc3_log'] = np.log(dataframe['hlc3'])
    dataframe['hl2_log'] = np.log(dataframe['hl2'])
    dataframe['ohlc4_log'] = np.log(dataframe['ohlc4'])

    dataframe['close_log'] = np.log(dataframe['close'])
    dataframe['high_log'] = np.log(dataframe['high'])
    dataframe['low_log'] = np.log(dataframe['low'])
    dataframe['open_log'] = np.log(dataframe['open'])
    return dataframe

class TM3Consumer(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)

    to make predictions on their data. populate_any_indicators() automatically
    generates the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """

    def heartbeat(self):
        sdnotify.SystemdNotifier().notify("WATCHDOG=1")

    def log(self, msg, *args, **kwargs):
        self.heartbeat()
        logger.info(msg, *args, **kwargs)

        plot_config = {
            "main_plot": {},
            "subplots": {
                "trend": {
                    "do_predict_tm3_1h": {
                        "color": "#102c42",
                        "type": "bar"
                        },
                    "trend_long_tm3_1h": {
                        "color": "#2db936",
                        "type": "line"
                        },
                    "trend_short_tm3_1h": {
                        "color": "#f40b5d",
                        "type": "line"
                        }
                    },
                "extrema": {
                    "do_predict_tm3_1h": {
                        "color": "#102c42",
                        "type": "bar"
                        },
                    "maxima_tm3_1h": {
                        "color": "#f40b5d",
                        "type": "line"
                        },
                    "minima_tm3_1h": {
                        "color": "#2db936"
                        }
                    }
                }
            }

    minimal_roi = {
        "0": 0.13
    }

    process_only_new_candles = False
    use_exit_signal = True
    can_short = True
    ignore_roi_if_entry_signal = True

    stoploss = -0.012
    trailing_stop = True
    trailing_only_offset_is_reached  = False
    trailing_stop_positive_offset = 0

    def add_slope_indicator(self, df: DataFrame, target_var = "ohlc4_log", predict_target = 6) -> DataFrame:
        df = df.set_index(df['date'])

        target = helpers.create_target(df, predict_target, method='polyfit', polyfit_var=target_var)
        target = target[['trend', 'slope', 'start_windows']].set_index('start_windows')
        target.fillna(0)

        # scale slope to 0-1
        target['slope'] = RobustScaler().fit_transform(target['slope'].values.reshape(-1, 1)).reshape(-1)

        target.rename(columns={'slope': f'{target_var}{predict_target}_exp_slope', 'trend': f'{target_var}{predict_target}_exp_trend'}, inplace=True)

        df = df.join(target[[f'{target_var}{predict_target}_exp_slope', f'{target_var}{predict_target}_exp_trend']], how='left')
        df = df.reset_index(drop=True)

        return df

    _columns_to_expect = [
        'do_predict',
        'minima',
        'maxima',
        'trend_long',
        'trend_short',
        'trend_strength',
        'trend_strength_abs',
        '&-trend_long_roc_auc',
        '&-trend_long_f1',
        '&-trend_long_logloss',
        '&-trend_long_accuracy',
        '&-trend_short_roc_auc',
        '&-trend_short_f1',
        '&-trend_short_logloss',
        '&-trend_short_accuracy',
        '&-extrema_maxima_roc_auc',
        '&-extrema_maxima_f1',
        '&-extrema_maxima_logloss',
        '&-extrema_maxima_accuracy',
        '&-extrema_minima_roc_auc',
        '&-extrema_minima_f1',
        '&-extrema_minima_logloss',
        '&-extrema_minima_accuracy'
    ]

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        self.log(f"ENTER .populate_indicators() {metadata} {df.shape}")
        start_time = time.time()

        pair = metadata['pair']
        timeframe = self.timeframe

        producer_pairs = self.dp.get_producer_pairs(producer_name="tm3_1h")

        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe, _ = self.dp.get_producer_df(pair, timeframe, producer_name="tm3_1h")

        if not producer_dataframe.empty:
            # print("producer dataframe")
            producer_dataframe = producer_dataframe[self._columns_to_expect + ['date']].copy()
            # print(producer_dataframe)
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            df = merge_informative_pair(df, producer_dataframe,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="tm3_1h")
        else:
            columns_to_expect_suffix = [col + "_tm3_1h" for col in self._columns_to_expect]
            df[columns_to_expect_suffix] = 0


        last_candle = df.iloc[-1].squeeze()
        # if not producer_dataframe.empty:
            # print("last candle")
            # print(last_candle)

        self.dp.send_msg(f"{metadata['pair']} predictions: \n  minima={last_candle['minima_tm3_1h']:.2f}, \n  maxima={last_candle['maxima_tm3_1h']:.2f}, \n  trend long={last_candle['trend_long_tm3_1h']:.2f}, \n  trend short={last_candle['trend_short_tm3_1h']:.2f}, \n  trend strength={last_candle['trend_strength_tm3_1h']:.2f}")


        self.log(f"EXIT populate_indicators {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df

    def signal_entry_long(self, df: DataFrame):
        general_condition = df['do_predict_tm3_1h'] == 1
        minima_condition1 = qtpylib.crossed_below(df['minima_tm3_1h'], 0.8) & (df['trend_short_tm3_1h'] < 0.6) # minima reached and trend is not short
        minima_condition2 = qtpylib.crossed_above(df['minima_tm3_1h'], 0.9) & (df['trend_short_tm3_1h'] < 0.7)
        # trend_condition = (df['trend_long'] >= 0.8) & (df['trend_strength_abs'] >= 0.4) & (df['maxima'] < 0.5) # trend is long and maxima is not reached
        # return minima_condition | trend_condition
        return general_condition & (minima_condition1 | minima_condition2)

    def signal_exit_long(self, df: DataFrame):
        general_condition = df['do_predict_tm3_1h'] == 1
        maxima_condition = df['maxima_tm3_1h'] >= 0.8
        # trend_condition = df['trend_long'] >= 0.9
        # return minima_condition | trend_condition
        return general_condition & maxima_condition


    def signal_entry_short(self, df: DataFrame):
        general_condition = df['do_predict_tm3_1h'] == 1
        maxima_condition1 = qtpylib.crossed_below(df['maxima_tm3_1h'], 0.8) & (df['trend_long_tm3_1h'] < 0.6) # maxima reached and trend is not long
        maxima_condition2 = qtpylib.crossed_above(df['maxima_tm3_1h'], 0.9) & (df['trend_long_tm3_1h'] < 0.7)
        # trend_condition = (df['trend_short'] >= 0.8) & (df['trend_strength_abs'] >= 0.4) & (df['minima'] < 0.5) # trend is short and minima is not reached

        # return maxima_condition | trend_condition
        return general_condition & (maxima_condition1 | maxima_condition2)

    def signal_exit_short(self, df: DataFrame):
        general_condition = df['do_predict_tm3_1h'] == 1
        maxima_condition = df['minima_tm3_1h'] >= 0.8
        # trend_condition = df['trend_long'] >= 0.9
        # return minima_condition | trend_condition
        return general_condition & maxima_condition


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[self.signal_entry_long(df), 'enter_long'] = 1

        df.loc[self.signal_entry_short(df),'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[self.signal_exit_long(df), 'exit_long'] = 1

        df.loc[self.signal_exit_short(df), 'exit_short'] = 1

        return df


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if df.empty:
            # Handle the empty DataFrame case
            return None  # Or other appropriate handling

        last_candle = df.iloc[-1].squeeze()

        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        is_short = trade.is_short == True
        is_long = trade.is_short == False
        is_profitable = current_profit > 0

        roi_result = self.check_roi(pair, current_time, trade.open_date_utc, current_profit)
        if roi_result:
            return roi_result

        if trade.is_open and is_long and last_candle['maxima_tm3_1h'] >= 0.6 and is_profitable:
            return "almost_maxima"

        if trade.is_open and is_long and last_candle['trend_short_tm3_1h'] >= 0.7 and is_profitable:
            return "trend_reserse_to_short"

        if trade.is_open and is_short and last_candle['minima_tm3_1h'] >= 0.6 and is_profitable:
            return "almost_minima"

        if trade.is_open and is_short and last_candle['trend_long_tm3_1h'] >= 0.7 and is_profitable:
            return "trend_reserse_to_long"


    ####
    # Dynamic ROI
    cached_roi_tables = {}

    def get_or_create_roi_table(self, pair, kernel=6):
        # Check cache first
        if pair in self.cached_roi_tables:
            return self.cached_roi_tables[pair]

        # Get analyzed dataframe
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None

        # Analyze candles to create ROI table
        min_peaks = argrelextrema(df["low"].values, np.less, order=kernel)[0]
        max_peaks = argrelextrema(df["high"].values, np.greater, order=kernel)[0]

        # Prepare lists for data
        distances = []
        candles_between_peaks = []

        # Iterate over low peaks to find the next high peak
        for low_peak in min_peaks:
            next_high_peaks = max_peaks[max_peaks > low_peak]
            if next_high_peaks.size > 0:
                high_peak = next_high_peaks[0]
                low_price = df.at[low_peak, 'close']
                high_price = df.at[high_peak, 'close']
                distance_percentage = ((high_price - low_price) / low_price)
                distances.append(distance_percentage)
                num_candles = high_peak - low_peak
                candles_between_peaks.append(num_candles)

        # Iterate over high peaks to find the next low peak
        for high_peak in max_peaks:
            next_low_peaks = min_peaks[min_peaks > high_peak]
            if next_low_peaks.size > 0:
                low_peak = next_low_peaks[0]
                high_price = df.at[high_peak, 'close']
                low_price = df.at[low_peak, 'close']
                distance_percentage = -((low_price - high_price) / high_price)
                distances.append(distance_percentage)
                num_candles = low_peak - high_peak
                candles_between_peaks.append(num_candles)

        if not distances or not candles_between_peaks:
            return None

        distances_description = pd.Series(distances).describe()
        candles_between_peaks_description = pd.Series(candles_between_peaks).describe()

        # Creating dynamic ROI table using calculated statistics
        dynamic_roi = {
            "0": distances_description['75%'],
            str(int(candles_between_peaks_description['25%'] * 60)): distances_description['50%'],
            str(int(candles_between_peaks_description['50%'] * 60)): distances_description['25%'],
            str(int(candles_between_peaks_description['75%'] * 60)): 0.00  # Using 75th percentile for the last tier
        }

        # Cache the ROI table
        self.cached_roi_tables[pair] = dynamic_roi
        return dynamic_roi


    def check_roi(self, pair, current_time, trade_open_date_utc, current_profit):
        dynamic_roi = self.get_or_create_roi_table(pair, kernel=self.TARGET_EXTREMA_KERNEL)
        if not dynamic_roi:
            return None

        # print("dymanic roi for pair:", pair)
        # print(dynamic_roi)

        trade_duration = (current_time - trade_open_date_utc).seconds / 60
        for roi_time, roi_value in dynamic_roi.items():
            if trade_duration >= int(roi_time) and current_profit >= roi_value:
                # print(f"ROI reached: {roi_value} at {roi_time} minutes")
                return "dynamic_roi"

        return None


    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, entry_tag: Optional[str],
    #                         side: str, **kwargs) -> bool:

    #     if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
    #         self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}, {entry_tag}, {side}")

    #     # if not enabled, exit with True
    #     if (not self.config['sagemaster'].get('enabled', False)):
    #         return True

    #     # get client and load params
    #     sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

    #     [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
    #     deal_type = 'buy' if side == 'long' else 'sell'
    #     tp_tip = round(self.LONG_TP.value * 100, 4) if side == 'long' else round(self.SHORT_TP.value * 100, 4)
    #     sl_tip = round(self.stoploss * 100, 4)

    #     # generate trade_id, which is +1 to last trade in db
    #     trade_id = "1"
    #     trade = Trade.get_trades(None).order_by(desc(Trade.open_date)).first()
    #     if (trade):
    #         trade_id = str(trade.id + 1)

    #     # convert trade_id to uuid
    #     trade_id = helpers.get_uuid_from_key(str(trade_id))

    #     sgm.open_deal(
    #         market=market,
    #         symbol_base=symbol_base,
    #         symbol_quote=symbol_quote,
    #         deal_type=deal_type,
    #         buy_price=rate,
    #         tp_tip=tp_tip,
    #         sl_tip=sl_tip,
    #         trade_id=trade_id
    #     )

    #     return True

    # def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
    #                        rate: float, time_in_force: str, exit_reason: str,
    #                        current_time: datetime, **kwargs) -> bool:

    #     if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
    #         self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}")

    #     # if not enabled, exit with True
    #     if (not self.config['sagemaster'].get('enabled', False)):
    #         return True

    #     sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

    #     [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
    #     tp_tip = round(self.LONG_TP.value * 100, 4) if trade.is_short == False else round(self.SHORT_TP.value * 100, 4)
    #     sl_tip = round(self.stoploss * 100, 4)
    #     profit_ratio = trade.calc_profit_ratio(rate)
    #     deal_type = 'buy' if trade.is_short == False else 'sell'
    #     trade_id = helpers.get_uuid_from_key(str(trade.id))
    #     allow_stoploss = self.config['sagemaster'].get('allow_stoploss', False)

    #     sgm.close_deal(
    #         market=market,
    #         symbol_base=symbol_base,
    #         symbol_quote=symbol_quote,
    #         deal_type=deal_type,
    #         buy_price=rate,
    #         tp_tip=tp_tip,
    #         sl_tip=sl_tip,
    #         trade_id=trade_id,
    #         profit_ratio=profit_ratio,
    #         allow_stoploss=allow_stoploss
    #     )

    #     return True
