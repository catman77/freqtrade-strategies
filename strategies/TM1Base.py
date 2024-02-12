# add common folders to path
import sys
import os

from freqtrade.enums.runmode import RunMode
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import sdnotify

from typing import Dict, List, Optional
from lib.ma import MovingAveragesCalculate, MovingAveragesCalculator2
from lib.mom import MomentumANDVolatilityCalculate
from lib.cycle import CycleCalculate
from lib.trend import TrendCalculate
from lib.oscillators import OscillatorsCalculate
from lib import helpers
from lib.sagemaster import SageMasterClient
from lib.Alpha101 import get_alpha
import lib.glassnode as gn
from scipy.special import softmax

import warnings
import json
import logging
from functools import reduce
import time
import numpy as np
from technical.pivots_points import pivots_points

import pandas as pd
from pandas import DataFrame
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import BooleanParameter, DecimalParameter, IntParameter
from datetime import timedelta, datetime, timezone
import multiprocessing as mp
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
import talib.abstract as ta
from sqlalchemy import desc

logger = logging.getLogger(__name__)

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_distance(p1, p2):
    return abs((p1) - (p2))

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


def f(x):
    return x


class TM1Base(IStrategy):
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

    class HyperOpt:
        def generate_estimator(dimensions: List['Dimension'], **kwargs):
            return "ET"
            # return "GP"
            # return "RF"

        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.04, -0.01, decimals=3, name='stoploss')]

        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(1, 180, name='roi_t1'),
                Integer(1, 180, name='roi_t2'),
                Integer(1, 180, name='roi_t3'),
                SKDecimal(0, 0.2, decimals=3, name='roi_p1'),
                SKDecimal(0, 0.06, decimals=3, name='roi_p2'),
                SKDecimal(0, 0.15, decimals=3, name='roi_p3'),
            ]

        def generate_roi_table(params: Dict) -> Dict[int, float]:

            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

            return roi_table

    minimal_roi = {"0": 100}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "signal": {
                "do_predict": {
                    "color": "#224116",
                    "type": "bar"
                },
                "trend_strength_abs": {
                    "color": "#5b1ca8",
                    "type": "line"
                },
                "trend_long": {
                    "color": "#47bf3b",
                    "type": "line"
                },
                "trend_short": {
                    "color": "#e84e6c",
                    "type": "line"
                }
            },
            "roc_auc": {
                "roc_auc_long_gini_12": {
                    "color": "#35e667",
                    "type": "line"
                },
                "roc_auc_short_gini_12": {
                    "color": "#cf2a8a",
                    "type": "line"
                },
                #  "roc_auc_long_gini_24": {
                #     "color": "#35e667",
                #     "type": "line"
                # },
                # "roc_auc_short_gini_24": {
                #     "color": "#cf2a8a",
                #     "type": "line"
                # }
            },
            "accuracy": {
                "accuracy_long_6": {
                    "color": "#35e667",
                    "type": "line"
                },
                "accuracy_short_6": {
                    "color": "#cf2a8a",
                    "type": "line"
                },
                # "accuracy_long_12": {
                #     "color": "#35e667",
                #     "type": "line"
                # },
                # "accuracy_short_12": {
                #     "color": "#cf2a8a",
                #     "type": "line"
                # },
                #  "accuracy_long_24": {
                #     "color": "#35e667",
                #     "type": "line"
                # },
                # "accuracy_short_24": {
                #     "color": "#cf2a8a",
                #     "type": "line"
                # }
            },
            "DI": {
                "DI_values": {
                    "color": "#3c51d7",
                    "type": "line"
                },
                "DI_cutoff": {
                    "color": "#99254a",
                    "type": "line"
                }
            }
        }
    }

    process_only_new_candles = True
    use_exit_signal = True
    can_short = True

    stoploss = -0.04
    trailing_stop = True
    trailing_only_offset_is_reached  = False
    trailing_stop_positive_offset = 0


    max_open_trades = 2

    TARGET_VAR = "ohlc4_log"
    DEBUG = False

    LONG_TP = DecimalParameter(0.005, 0.03, decimals=4, default=0.0236, space="sell", optimize=False)
    SHORT_TP = DecimalParameter(0.005, 0.03, decimals=4, default=0.0162, space="sell", optimize=False)

    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    # internally freqtrade multiply it by 2, so we put here 1/2 of the max startup candle count
    startup_candle_count: int = 100

    @property
    def PREDICT_TARGET(self):
        return self.config["freqai"].get("label_period_candles", 6)

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 12,
                "required_profit": 0.0,
                "only_per_pair": True,
                "only_per_side": True
            }
        ]

    def bot_start(self, **kwargs) -> None:
        print("bot_start")

        self.DEBUG = self.config["sagemaster"].get("debug", False)

    def feature_engineering_candle_patterns(self, df, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_candle_patterns(): {metadata} {df.shape}")
        start_time = time.time()
        pref = f"%"
        df = df.copy()

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # Hammer: values [0, 100]
        df[f'{pref}-CDLHAMMER'] = ta.CDLHAMMER(df)
        # Inverted Hammer: values [0, 100]
        df[f'{pref}-CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df)
        # Dragonfly Doji: values [0, 100]
        df[f'{pref}-CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df)
        # Piercing Line: values [0, 100]
        df[f'{pref}-CDLPIERCING'] = ta.CDLPIERCING(df) # values [0, 100]
        # Morningstar: values [0, 100]
        df[f'{pref}-CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        df[f'{pref}-CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        df[f'{pref}-CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df)
        # Shooting Star: values [0, 100]
        df[f'{pref}-CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df)
        # Gravestone Doji: values [0, 100]
        df[f'{pref}-CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df)
        # Dark Cloud Cover: values [0, 100]
        df[f'{pref}-CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df)
        # Evening Doji Star: values [0, 100]
        df[f'{pref}-CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df)
        # Evening Star: values [0, 100]
        df[f'{pref}-CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # Three Line Strike: values [0, -100, 100]
        df[f'{pref}-CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df)
        # Spinning Top: values [0, -100, 100]
        df[f'{pref}-CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        df[f'{pref}-CDLENGULFING'] = ta.CDLENGULFING(df) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        df[f'{pref}-CDLHARAMI'] = ta.CDLHARAMI(df) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        df[f'{pref}-CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        df[f'{pref}-CDL3INSIDE'] = ta.CDL3INSIDE(df) # values [0, -100, 100]

        # Pattern Recognition - The Rest
        # ------------------------------------
        # df[f'{pref}-CDL2CROWS'] = ta.CDL2CROWS(df)
        # df[f'{pref}-CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df)
        # df[f'{pref}-CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df)
        # df[f'{pref}-CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df)
        # df[f'{pref}-CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df)
        # df[f'{pref}-CDLBELTHOLD'] = ta.CDLBELTHOLD(df)
        # df[f'{pref}-CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df)
        # df[f'{pref}-CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df)
        # df[f'{pref}-CDLDOJI'] = ta.CDLDOJI(df)
        # df[f'{pref}-CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df)
        # df[f'{pref}-CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df)
        # df[f'{pref}-CDLHIKKAKE'] = ta.CDLHIKKAKE(df)
        # df[f'{pref}-CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df)
        # df[f'{pref}-CDLINNECK'] = ta.CDLINNECK(df)
        # df[f'{pref}-CDLKICKING'] = ta.CDLKICKING(df)
        # df[f'{pref}-CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df)
        # df[f'{pref}-CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df)
        # df[f'{pref}-CDLLONGLINE'] = ta.CDLLONGLINE(df)
        # df[f'{pref}-CDLMARUBOZU'] = ta.CDLMARUBOZU(df)
        # df[f'{pref}-CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df)
        # df[f'{pref}-CDLMATHOLD'] = ta.CDLMATHOLD(df)
        # df[f'{pref}-CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df)
        # df[f'{pref}-CDLONNECK'] = ta.CDLONNECK(df)
        # df[f'{pref}-CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df)
        # df[f'{pref}-CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df)
        # df[f'{pref}-CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df)
        # df[f'{pref}-CDLSHORTLINE'] = ta.CDLSHORTLINE(df)
        # df[f'{pref}-CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df)
        # df[f'{pref}-CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df)
        # df[f'{pref}-CDLTAKURI'] = ta.CDLTAKURI(df)
        # df[f'{pref}-CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df)
        # df[f'{pref}-CDLTHRUSTING'] = ta.CDLTHRUSTING(df)
        # df[f'{pref}-CDLTRISTAR'] = ta.CDLTRISTAR(df)
        # df[f'{pref}-CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df)
        # df[f'{pref}-CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df)

        self.log(f"EXIT .feature_engineering_candle_patterns() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df.copy()

    def new_pool(self):
        return mp.Pool(self.config["freqai"].get("data_kitchen_thread_count", 4))

    def feature_engineering_alphas101(self, df:DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_alphas101(): {metadata} {df.shape}")
        start_time = time.time()

        df = get_alpha(df)

        self.log(f"EXIT .feature_engineering_alphas101() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def feature_engineering_glassnode2(self, df:DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_glassnode2(): {metadata} {df.shape}")
        start_time = time.time()
        the_pool = self.new_pool()

        # load tasks
        gn_tasks = json.loads(open('user_data/valid_glassnode_tasks.json', 'r').read())

        # print("before glassnode join")
        # print(df)

        gn_jobs = []
        gn_results = []
        # iterate over resolutions
        for resolution in ['1h', '24h']:
            gn_tasks_filtered = list(filter(lambda x: x['resolution'] == resolution, gn_tasks))

            max_timestamp = int(df['date'].max().timestamp())
            if (resolution == '24h'):
                # if 24h, set the max timestamp to the end of the day
                max_timestamp = int(datetime.fromtimestamp(max_timestamp, tz=timezone.utc).replace(hour=23, minute=59, second=59).timestamp())
            elif (resolution == '1h'):
                # round to nearest hour
                max_timestamp = int(datetime.fromtimestamp(max_timestamp, tz=timezone.utc).replace(minute=59, second=59).timestamp())

            df_min_timestamp = int(df['date'].min().timestamp() - 86500)
            # calculate min date based on train period days
            train_min_timestamp = df_min_timestamp - ((self.config["freqai"]["train_period_days"] + 25) * 24 * 3600)

            self.log(f"GLASSNODE resolution: {resolution} min_timestamp: {train_min_timestamp}, max_timestamp: {max_timestamp} tasks: {len(gn_tasks_filtered)}")

            for t in gn_tasks_filtered:
                gn_jobs.append(the_pool.apply_async(gn.process_task, (t, train_min_timestamp, max_timestamp)))

        for r in gn_jobs:
            gn_results.append(r.get())

        gn_df = pd.concat(gn_results, axis=1)
        gn_df = gn_df.fillna(method='ffill')
        gn_df.fillna(gn_df.mean(), inplace=True)
        # print("after glassnode join")
        # print(gn_df)

        df = df.set_index(df['date'])
        df = df.join(gn_df, how='left')
        df = df.reset_index(drop=True)

        # in case we are on lower df then 1h, we need to forward fill the data in df
        df = df.fillna(method='ffill')

        # df after merge
        # print(df)
        # print(df.tail(32))

        # print("glassnode fetching done.")

        self.log(f"EXIT .feature_engineering_glassnode2() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def feature_engineering_glassnode(self, df, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_glassnode(): {metadata} {df.shape}")
        start_time = time.time()

        # find min and max timestamps from df
        gn_tasks = json.loads(open('user_data/valid_glassnode_tasks.json', 'r').read())

        # fina max date from df
        max_timestamp = int(df['date'].max().timestamp() + 3600)
        # min timestamp from df
        df_min_timestamp = int(df['date'].min().timestamp())
        # calculate min date based on train period days
        train_min_timestamp = df_min_timestamp - ((self.config["freqai"]["train_period_days"] + 25) * 24 * 3600)

        # print(f"GLASSNODE min_timestamp: {train_min_timestamp}, max_timestamp: {max_timestamp} tasks: {len(gn_tasks)} df: {df.shape}")

        gn_metrics = map(lambda x: gn.process_task(x, train_min_timestamp, max_timestamp), gn_tasks)
        gn_df = pd.concat(gn_metrics, axis=1)

        df = df.set_index(df['date'])
        df = df.join(gn_df, how='left')
        df = df.reset_index(drop=True)

        # print("glassnode fetching done.")

        self.log(f"EXIT .feature_engineering_glassnode() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def feature_engineering_trend(self, df: DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_trend() {metadata} {df.shape}")
        start_time = time.time()

        the_pool = self.new_pool()

        # Trends for indicators
        all_cols = filter(lambda col:
            (col != 'trend')
            and col.find('pmX') == -1
            and col.find('date') == -1
            and col.find('_signal') == -1
            and col.find('_trend') == -1
            and col.find('_rising') == -1
            and col.find('_std') == -1
            and col.find('_change_') == -1
            and col.find('_lower_band') == -1
            and col.find('_upper_band') == -1
            and col.find('_upper_envelope') == -1
            and col.find('_lower_envelope') == -1
            and col.find('%-dist_to_') == -1
            and col.find('%-s1') == -1
            and col.find('%-s2') == -1
            and col.find('%-s3') == -1
            and col.find('%-r1') == -1
            and col.find('%-r2') == -1
            and col.find('%-r3') == -1
            and col.find('_divergence') == -1, df.columns)

        results = []
        result_cols = []
        # launch all processes
        for col in all_cols:
            result = the_pool.apply_async(helpers.create_col_trend, (col, self.PREDICT_TARGET, df, "polyfit"))
            results.append(result)

        # collect all results
        for result in results:
            result_cols.append(result.get())

        df = pd.concat([df, *result_cols], axis=1)

        self.log(f"EXIT .feature_engineering_trend() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df

    def feature_engineering_expand_basic(self, df, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_expand_basic() {metadata} {df.shape}")
        start_time = time.time()

        # log data
        df = candle_stats(df)

        # add Alpha101
        df = self.feature_engineering_alphas101(df, metadata, **kwargs)

        # add TA features to dataframe

        # Moving Averages
        mac = MovingAveragesCalculator2(col_prefix="%-mac-", col_target='ohlc4_log', config = {
                "SMA": [24, 48, 96, 192],
                "EMA": [12, 24, 48, 96],
                "HMA": [12, 24, 48, 96],
                # "JMA": [6, 12, 24, 48],
                "KAMA": [12, 24, 48, 96],
                "ZLSMA": [12, 24, 48, 96],
            })

        df = mac.calculate_moving_averages(df)

        # Momentum Indicators
        mvc = MomentumANDVolatilityCalculate(
            df, open_col = 'open_log', close_col='close_log', high_col='high_log', low_col='low_log')
        df = mvc.calculate_all().copy()

        # Cycle Indicators
        cc = CycleCalculate(df, calc_col='close_log')
        df = cc.calculate_all().copy()

        # Trend Indicators
        tc = TrendCalculate(df, close_col='close_log', high_col='high_log', low_col='low_log',
                            open_col='open_log')
        df = tc.calculate_all().copy()

        # Oscillator Indicators
        oc = OscillatorsCalculate(df, close_col='close_log')
        df = oc.calculate_all().copy()

        # Pivot Points
        pp = pivots_points(df, timeperiod=100)
        df['r1'] = pp['r1']
        df['s1'] = pp['s1']
        df['r2'] = pp['r2']
        df['s2'] = pp['s2']
        df['r3'] = pp['r3']
        df['s3'] = pp['s3']

        df['%-dist_to_r1'] = get_distance(df['close'], df['r1'])
        df['%-dist_to_r2'] = get_distance(df['close'], df['r2'])
        df['%-dist_to_r3'] = get_distance(df['close'], df['r3'])
        df['%-dist_to_s1'] = get_distance(df['close'], df['s1'])
        df['%-dist_to_s2'] = get_distance(df['close'], df['s2'])
        df['%-dist_to_s3'] = get_distance(df['close'], df['s3'])

        cat_col = [x for x in df if x.find('pmX_10_3_12_1') != -1]

        for col in cat_col:
            df[col] = df[col].map({'down': 0, 'up': 1})

        # rename generated features so freqtrade can recognize them
        for col in df.columns:
            if col.startswith('%-') or col in ['date', 'volume', 'hl2_log', 'hl2', 'hlc3', 'hlc3_log', 'ohlc4', 'ohlc4_log', 'open', 'high', 'low', 'close', 'low_log', 'open_log', 'high_log', 'low_log', 'close_log']:
                continue
            else:
                df.rename(columns={col: "%-" + col}, inplace=True)

        # defragment df
        df = df.copy()

        # calculate trend for all features
        df = self.feature_engineering_trend(df, metadata, **kwargs).copy()

        # add chart pattern features
        df = self.feature_engineering_candle_patterns(df, metadata, **kwargs).copy()

        self.log(f"EXIT .feature_engineering_expand_basic() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df

    def feature_engineering_standard(self, df: DataFrame, metadata, **kwargs):
        self.log(f"ENTER .feature_engineering_standard() {metadata} {df.shape}")
        start_time = time.time()

        # add lag
        df = helpers.create_lag(df, 6)

        # glassnode is disabled for now
        # if not self.DEBUG:
            # Import Glassnode data
            # df = self.feature_engineering_glassnode2(df, metadata)

        # some basic features
        df["%-pct-change"] = df["close"].pct_change()
        df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
        df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

        # volume features
        df["%-volume"] = df["volume"].copy()
        df = gn.extract_feature_metrics(df, "%-volume")
        df = df.copy()

        # fill empty values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        # fill empty columns with 0
        df = df.fillna(0).copy()

        self.log(f"EXIT .feature_engineering_standard() {metadata} {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def label_profit_filter(self, df: DataFrame, perc_threshold):
        # Reverse dataframe
        df = df[::-1]

        # Calculate highest and lowest values
        df['high_6'] = df['high'].rolling(6).max() / df['close'] - 1
        df['low_6'] = df['low'].rolling(6).min() / df['close'] - 1

        # Reverse it back
        df = df[::-1]

        # Filter out weak changes
        df.loc[df['high_6'] < perc_threshold, '&-trend_long'] = 'trend_not_long'
        df.loc[df['low_6'] > -perc_threshold, '&-trend_short'] = 'trend_not_short'

    def set_freqai_targets(self, df: DataFrame, metadata, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        self.log(f"ENTER .set_freqai_targets() {metadata} {df.shape}")
        start_time = time.time()

        df = candle_stats(df)

        df = df.set_index(df['date'])
        target = helpers.create_target(df, self.PREDICT_TARGET,
                                       method='polyfit', polyfit_var=self.TARGET_VAR)
        target = target[['trend', 'start_windows']].set_index('start_windows')

        target['&-trend_long'] = np.where(target['trend'] == 1, 'trend_long', 'trend_not_long')
        target['&-trend_short'] = np.where(target['trend'] == -1, 'trend_short', 'trend_not_short')

        df = pd.concat([df, target[['&-trend_long', '&-trend_short']].copy()], axis=1)
        df = df.reset_index(drop=True)

        # EXPERIMENT: set labels only to signals which lead to 1% change
        # self.label_profit_filter(df, 0.01)

        # remove duplicated columns
        df = df.loc[:,~df.columns.duplicated(keep='first')]

        # save dataframe to csv with timestamp
        # dataframe.to_csv("freqai_data_{}.csv".format(int(time.time())))

        # make great again
        df = df.drop(columns=[
            'open_log', 'hlc3', 'ohlc4', 'low_log', 'close_log', 'high_log', 'ohlc4_log', 'hl2_log', 'hl2', 'hlc3_log'
        ])

        self.log(f"EXIT .set_freqai_targets() {df.shape}, execution time: {time.time() - start_time:.2f} seconds")

        return df


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        self.log(f"ENTER .populate_indicators() {metadata} {df.shape}")
        start_time = time.time()

        df = self.freqai.start(df, metadata, self)

        # trend strength indicator
        df['trend_strength'] = df['trend_long'] - df['trend_short']
        df['trend_strength_abs'] = abs(df['trend_strength'])
        df['trend_short_inverse'] = df['trend_short'] * -1

        # calculate softmax probabilities for trend
        # df['trend_long_softmax'] = np.exp(df['trend_long']) / (np.exp(df['trend_long']) + np.exp(df['trend_short']))
        # df['trend_short_softmax'] = np.exp(df['trend_short']) / (np.exp(df['trend_long']) + np.exp(df['trend_short']))

        df['L1'] = 1.0
        df['L0'] = 0
        df['L-1'] = -1.0

        # save df to file
        # df.to_csv("df_{}.csv".format(int(time.time())))

        self.log(f"EXIT populate_indicators {df.shape}, execution time: {time.time() - start_time:.2f} seconds")
        return df


    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:

        if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
            self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}, {entry_tag}, {side}")

        # if not enabled, exit with True
        if (not self.config['sagemaster'].get('enabled', False)):
            return True

        # get client and load params
        sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

        [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
        deal_type = 'buy' if side == 'long' else 'sell'
        tp_tip = round(self.LONG_TP.value * 100, 4) if side == 'long' else round(self.SHORT_TP.value * 100, 4)
        sl_tip = round(self.stoploss * 100, 4)

        # generate trade_id, which is +1 to last trade in db
        trade_id = "1"
        trade = Trade.get_trades(None).order_by(desc(Trade.open_date)).first()
        if (trade):
            trade_id = str(trade.id + 1)

        # convert trade_id to uuid
        trade_id = helpers.get_uuid_from_key(str(trade_id))

        sgm.open_deal(
            market=market,
            symbol_base=symbol_base,
            symbol_quote=symbol_quote,
            deal_type=deal_type,
            buy_price=rate,
            tp_tip=tp_tip,
            sl_tip=sl_tip,
            trade_id=trade_id
        )

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        if self.config.get('runmode') in (RunMode.DRY_RUN, RunMode.LIVE):
            self.log(f"ENTER confirm_trade_entry() {pair}, {current_time}, {rate}")

        # if not enabled, exit with True
        if (not self.config['sagemaster'].get('enabled', False)):
            return True

        sgm = SageMasterClient(self.config['sagemaster']['webhook_api_key'], self.config['sagemaster']['webhook_url'], self.config['sagemaster']['trader_nickname'])

        [market, symbol_base, symbol_quote] = helpers.extract_currencies(pair)
        tp_tip = round(self.LONG_TP.value * 100, 4) if trade.is_short == False else round(self.SHORT_TP.value * 100, 4)
        sl_tip = round(self.stoploss * 100, 4)
        profit_ratio = trade.calc_profit_ratio(rate)
        deal_type = 'buy' if trade.is_short == False else 'sell'
        trade_id = helpers.get_uuid_from_key(str(trade.id))
        allow_stoploss = self.config['sagemaster'].get('allow_stoploss', False)

        sgm.close_deal(
            market=market,
            symbol_base=symbol_base,
            symbol_quote=symbol_quote,
            deal_type=deal_type,
            buy_price=rate,
            tp_tip=tp_tip,
            sl_tip=sl_tip,
            trade_id=trade_id,
            profit_ratio=profit_ratio,
            allow_stoploss=allow_stoploss
        )

        return True