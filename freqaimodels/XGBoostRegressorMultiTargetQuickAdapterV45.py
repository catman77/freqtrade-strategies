import logging
from typing import Any, Dict

from xgboost import XGBRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
# from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import scipy as spy
import numpy.typing as npt
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import random
import optuna
# import warnings
from freqtrade.freqai.tensorboard import TBCallback
import sklearn
from optuna.samplers import TPESampler
from scipy.signal import argrelextrema
from freqtrade.freqai.utils import get_tb_logger, plot_feature_importance
from freqtrade.configuration import TimeRange
from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_seconds

logger = logging.getLogger(__name__)


# Optuna
N_TRIALS = 25

"""
The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
We use sponsor money to help stimulate new features and to pay for running these public
experiments, with a an objective of helping the community make smarter choices in their
ML journey.

This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
returns. The goal is to demonstrate gratitude to people who support the project and to
help them find a good starting point for their own creativity.

If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

https://github.com/sponsors/robcaulk
"""


class XGBoostRegressorMultiTargetQuickAdapterV45(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"],
                         data_dictionary["test_labels"])]
            eval_weights = [data_dictionary['test_weights']]

        sample_weight = data_dictionary["train_weights"]
        start = time.time()
        xgb_model = self.get_init_model(dk.pair)

        model = XGBRegressor(**self.model_training_parameters)

        model.set_params(callbacks=[TBCallback(
            dk.data_path)], activate=self.activate_tensorboard)
        model.fit(X=X, y=y, sample_weight=sample_weight, eval_set=eval_set,
                  sample_weight_eval_set=eval_weights, xgb_model=xgb_model)
        # set the callbacks to empty so that we can serialize to disk later
        model.set_params(callbacks=[])
        time_spent = (time.time() - start)
        self.dd.update_metric_tracker('fit_time', time_spent, dk.pair)

        return model, eval_set

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:

        warmed_up = True
        num_candles = self.freqai_info.get('fit_live_predictions_candles', 100)
        if self.live:
            if not hasattr(self, 'exchange_candles'):
                self.exchange_candles = len(
                    self.dd.model_return_values[pair].index)
            candle_diff = len(self.dd.historic_predictions[pair].index) - \
                (num_candles + self.exchange_candles)
            if candle_diff < 0:
                logger.warning(
                    f'Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go')
                warmed_up = False

        pred_df_full = self.dd.historic_predictions[pair].tail(
            num_candles).reset_index(drop=True)
        pred_df_sorted = pd.DataFrame()
        for label in pred_df_full.keys():
            if pred_df_full[label].dtype == object:
                continue
            pred_df_sorted[label] = pred_df_full[label]

        # pred_df_sorted = pred_df_sorted
        for col in pred_df_sorted:
            pred_df_sorted[col] = pred_df_sorted[col].sort_values(
                ascending=False, ignore_index=True)
        frequency = num_candles / \
            (self.freqai_info['feature_parameters']
             ['label_period_candles'] * 2)
        max_pred = pred_df_sorted.iloc[:int(frequency)].mean()
        min_pred = pred_df_sorted.iloc[-int(frequency):].mean()

        if not warmed_up:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = 2
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = -2
        else:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = max_pred['&s-extrema']
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = min_pred['&s-extrema']

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for ft in dk.label_list:
            # f = spy.stats.norm.fit(pred_df_full[ft])
            dk.data['labels_std'][ft] = 0  # f[1]
            dk.data['labels_mean'][ft] = 0  # f[0]

        # fit the DI_threshold
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            f = spy.stats.weibull_min.fit(pred_df_full['DI_values'])
            cutoff = spy.stats.weibull_min.ppf(0.999, *f)

        dk.data["DI_value_mean"] = pred_df_full['DI_values'].mean()
        dk.data["DI_value_std"] = pred_df_full['DI_values'].std()
        dk.data['extra_returns_per_train']['DI_value_param1'] = f[0]
        dk.data['extra_returns_per_train']['DI_value_param2'] = f[1]
        dk.data['extra_returns_per_train']['DI_value_param3'] = f[2]
        dk.data['extra_returns_per_train']['DI_cutoff'] = cutoff

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, window: int, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(
            f"-------------------- Starting training {pair} --------------------")

        start_time = time.time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date} --------------------")
        # split data into train/test data.
        dd = self.make_train_test_datasets(features_filtered, labels_filtered, dk)

        # chop to the current window
        dd["train_features"] = dd["train_features"][-window:]
        dd["train_labels"] = dd["train_labels"][-window:]
        dd["train_weights"] = dd["train_weights"][-window:]

        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()

        # optional additional data cleaning/analysis
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])
        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (dd["test_features"],
             dd["test_labels"],
             dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                                 dd["test_labels"],
                                                                 dd["test_weights"])
            dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(
            f"Training model on {len(dd['train_features'])} data points")

        model, eval_set = self.fit(dd, dk)

        end_time = time.time()

        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")

        return model, eval_set

    def balance_training_weights(self, labels: DataFrame, weights: npt.ArrayLike, dk: FreqaiDataKitchen) -> npt.ArrayLike:
        """
        Modify training weights to emphasize unbalanced target labels, i.e., when one "class" (not
        exclusive to classification targets) is more numerous than the other.
        """
        label = dk.label_list[0]
        logger.info(f"using {label} to balance the weights")
        balance_weights = labels[label].abs().values.ravel()
        weights_balanced = weights + balance_weights
        scaled_weights = (weights_balanced - weights_balanced.min()) / \
            (weights_balanced.max() - weights_balanced.min())
        return scaled_weights

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame, dk: FreqaiDataKitchen
    ) -> Dict[Any, Any]:
        """
        Given the dataframe for the full history for training, split the data into
        training and test data according to user specified parameters in configuration
        file.
        :param filtered_dataframe: cleaned dataframe ready to be split.
        :param labels: cleaned labels ready to be split.
        """
        feat_dict = dk.freqai_config["feature_parameters"]

        if 'shuffle' not in dk.freqai_config['data_split_parameters']:
            dk.freqai_config["data_split_parameters"].update(
                {'shuffle': False})

        weights: npt.ArrayLike
        if feat_dict.get("weight_factor", 0) > 0:
            weights = dk.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

        if feat_dict.get("balance_weights", False):
            weights = self.balance_training_weights(labels, weights, dk)

        if dk.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe[: filtered_dataframe.shape[0]],
                labels,
                weights,
                **dk.config["freqai"]["data_split_parameters"],
            )
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights

        if feat_dict["shuffle_after_split"]:
            rint1 = random.randint(0, 100)
            rint2 = random.randint(0, 100)
            train_features = train_features.sample(
                frac=1, random_state=rint1).reset_index(drop=True)
            train_labels = train_labels.sample(
                frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = pd.DataFrame(train_weights).sample(
                frac=1, random_state=rint1).reset_index(drop=True).to_numpy()[:, 0]
            test_features = test_features.sample(
                frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(
                frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = pd.DataFrame(test_weights).sample(
                frac=1, random_state=rint2).reset_index(drop=True).to_numpy()[:, 0]

        # Simplest way to reverse the order of training and test data:
        if dk.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return dk.build_data_dictionary(
                test_features, train_features, test_labels,
                train_labels, test_weights, train_weights
            )
        else:
            return dk.build_data_dictionary(
                train_features, test_features, train_labels,
                test_labels, train_weights, test_weights
            )

    def extract_data_and_train_model(
        self,
        new_trained_timerange: TimeRange,
        pair: str,
        strategy: IStrategy,
        dk: FreqaiDataKitchen,
        data_load_timerange: TimeRange,
    ):
        """
        Retrieve data and train model.
        :param new_trained_timerange: TimeRange = the timerange to train the model on
        :param metadata: dict = strategy provided metadata
        :param strategy: IStrategy = user defined strategy object
        :param dk: FreqaiDataKitchen = non-persistent data container for current coin/loop
        :param data_load_timerange: TimeRange = the amount of data to be loaded
                                    for populating indicators
                                    (larger than new_trained_timerange so that
                                    new_trained_timerange does not contain any NaNs)
        """

        corr_dataframes, base_dataframes = self.dd.get_base_and_corr_dataframes(
            data_load_timerange, pair, dk
        )

        unfiltered_dataframe = dk.use_strategy_to_populate_indicators(
            strategy, corr_dataframes, base_dataframes, pair
        )

        trained_timestamp = new_trained_timerange.stopts

        self.tb_logger = get_tb_logger(self.dd.model_type, dk.data_path,
                                       self.activate_tensorboard)

        # add optuna error as None to dk so we can track it in the objective
        dk.optuna_error = None
        hp = {}
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, unfiltered_dataframe, pair, dk, new_trained_timerange),
            n_trials=N_TRIALS,
            n_jobs=1,
        )

        # display params
        hp = study.best_params
        # trial = study.best_trial
        for key, value in hp.items():
            logger.warning(f"{key:>20s} : {value}")
        logger.info(f"{'best objective value':>20s} : {study.best_value}")

        model = dk.best_model
        self.tb_logger.close()

        # incase we want to use this in the strat (but we need to be careful
        # training may update this while a trade may be on an older kernel)
        self.dd.pair_dict[pair]["kernel"] = hp["kernel"]
        self.dd.pair_dict[pair]["trained_timestamp"] = trained_timestamp
        dk.set_new_model_names(pair, trained_timestamp)
        self.dd.save_data(model, pair, dk)

        if self.plot_features:
            plot_feature_importance(model, pair, dk, self.plot_features)

        self.dd.purge_old_models()

    def set_freqai_targets(self, dataframe, kernel, **kwargs):
        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=kernel
        )
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
            order=kernel
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima-exit"] = np.where(
            dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima-exit"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)

        # predict the expected range
        dataframe['&-s_max'] = dataframe["close"].shift(-kernel).rolling(
            kernel).max()/dataframe["close"] - 1
        dataframe['&-s_min'] = dataframe["close"].shift(-kernel).rolling(
            kernel).min()/dataframe["close"] - 1

        return dataframe

    def objective(self, trial, unfiltered_dataframe, pair, dk, timerange):
        """Define the objective function"""

        window = trial.suggest_int('train_period_candles', 1152, 17280, step=2016)
        kernel = trial.suggest_int('kernel', 10, 50, step=10)

        opt_dataframe = self.set_freqai_targets(unfiltered_dataframe, kernel)

        timerange.stopts -= kernel * timeframe_to_seconds(self.config["timeframe"])
        timerange.startts += kernel * timeframe_to_seconds(self.config["timeframe"])

        unfiltered_dataframe = dk.slice_dataframe(timerange, unfiltered_dataframe)

        # find the features indicated by strategy and store in datakitchen
        dk.find_features(unfiltered_dataframe)
        dk.find_labels(unfiltered_dataframe)

        model, eval_set = self.train(opt_dataframe, pair, dk, window)
        X_test = eval_set[0][0]
        y_test = eval_set[0][1]

        y_pred = model.predict(X_test)

        error = sklearn.metrics.mean_squared_error(y_test, y_pred)
        if dk.optuna_error is None or error < dk.optuna_error:
            dk.optuna_error = error
            dk.best_model = model

        return error
