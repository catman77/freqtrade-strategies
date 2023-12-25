from datetime import datetime, timedelta
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict
import sdnotify

from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from typing import Any, Dict, Tuple
from datasieve.transforms import SKLearnWrapper
from datasieve.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import datasieve.transforms as ds
import wandb
from freqaimodels.MultiOutputClassifierWithFeatureSelect import MultiOutputClassifierWithFeatureSelect
import pytz

logger = logging.getLogger(__name__)

def heartbeat():
    sdnotify.SystemdNotifier().notify("WATCHDOG=1")

def log(msg, *args, **kwargs):
    heartbeat()
    logger.info(msg, *args, **kwargs)


class CatboostFeatureSelectMultiTargetBinaryClassifierV2(BaseClassifierModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    FEATURE_SELECT_MODEL_CONFIG = {
            "task_type": "CPU",
            "eval_metric": 'AUC',
            "auto_class_weights": 'Balanced',
            "colsample_bylevel": 0.096,
            "depth": 4,
            "boosting_type": "Plain",
            "bootstrap_type": "MVS",
            "l2_leaf_reg": 5.0,
            "learning_rate": 0.09,
            "save_snapshot": False,
            "allow_writing_files": False,
            "random_seed": 42,
        }

    @property
    def SELECT_FEATURES_ITERATIONS(self):
        return self.config["sagemaster"].get("CATBOOST_SELECT_FEATURES_ITERATIONS", 100)

    @property
    def NUM_FEATURES_TO_SELECT(self):
        return  self.config["sagemaster"].get("CATBOOST_NUM_FEATURES_TO_SELECT", 1024)

    @property
    def SELECT_FEATURES_STEPS(self):
        return self.config["sagemaster"].get("CATBOOST_SELECT_FEATURES_STEPS", 10)

    @property
    def AUTODETECT_NUM_FEATURES_TO_SELECT(self):
        return self.config["sagemaster"].get("CATBOOST_AUTODETECT_NUM_FEATURES_TO_SELECT", False)

    @property
    def FEATURE_SELECT_LABEL(self):
        return self.config["sagemaster"].get("CATBOOST_FEATURE_SELECT_LABEL", "&-trend_long")

    @property
    def FEATURE_SELECT_PERIOD_DAYS(self):
        return self.config["sagemaster"].get("CATBOOST_FEATURE_SELECT_PERIOD_DAYS", 5)


    @property
    def MODEL_IDENTIFIER(self):
        return self.config["freqai"].get("identifier", "default")

    def wandb_init(self, project:str = "TM3", name: str = None, job_type: str = None, config: Dict = None):
        # wandb.login(key=["ca8202923f1f937fac88fd39668ab6c97ec7d808"])
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            job_type=job_type,
            name=name + " " + datetime.now().strftime("%Y%m%d_%H%M%S"),

            # track hyperparameters and run metadata
            config=config
        )


    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """
        # select features
        selected_features_all_labels = self.feature_select(data_dictionary, dk)
        labels = list(selected_features_all_labels.keys())

        # selected_features = data_dictionary["train_features"].columns.tolist()
        dk.data['selected_features'] = selected_features_all_labels

        X = data_dictionary["train_features"].copy()
        y = data_dictionary["train_labels"]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            eval_sets = [None] * data_dictionary['test_labels'].shape[1]

            for i, label in enumerate(data_dictionary['test_labels'].columns):
                eval_sets[i] = Pool(
                    data=data_dictionary["test_features"][selected_features_all_labels[label]].copy(),
                    label=data_dictionary["test_labels"][label],
                    # weight=data_dictionary["test_weights"],
                )

        self.wandb_init(name=f"{self.MODEL_IDENTIFIER}_{dk.pair}.fit",
                        job_type="fit",
                        config=self.model_training_parameters)

        # train final model
        estimator = CatBoostClassifier(
            allow_writing_files=True,
            train_dir=Path(dk.data_path),
            **self.model_training_parameters,
        )

        init_models = [None] * y.shape[1]
        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append({
                    'eval_set': eval_sets[i],  'init_model': init_models[i],
                    'log_cout': sys.stdout, 'log_cerr': sys.stderr,
                    'callbacks': [wandb.catboost.WandbCallback()],
                 })

        multi_model = MultiOutputClassifierWithFeatureSelect(estimator=estimator)
        multi_model.fit(X=X, y=y, sample_weight=data_dictionary["train_weights"], selected_features_all_labels=selected_features_all_labels, fit_params=fit_params)

        # for i, estim in enumerate(multi_model.estimators_):
            # wandb.catboost.plot_feature_importances(estim, labels[i])

        wandb.finish()

        return multi_model

    def get_cache_filename(self, dk, date):
        # Representing the data path as a Path object
        data_path = Path(dk.data_path)

        # Navigate one folder up and then into the 'selected_features' folder
        selected_features_folder = data_path.parent / 'selected_features'

        # Create the folder if it does not exist
        selected_features_folder.mkdir(parents=True, exist_ok=True)

        # Return the complete path for the cache file
        return os.path.join(selected_features_folder, f'selected_features_{date.strftime("%Y%m%d")}.pkl')

    def load_cached_features(self, dk, date):
        # List all cache files in the 'selected_features' directory
        cache_files = list(Path(dk.data_path).parent.glob('selected_features/*.pkl'))

        # Convert to UTC timezone if date is timezone-aware
        utc = pytz.UTC if date.tzinfo is not None and date.tzinfo.utcoffset(date) is not None else None

        valid_files = []
        for f in cache_files:
            # Extracting date from filename
            file_date_str = f.stem.split('_')[-1]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            if utc:
                file_date = utc.localize(file_date)

            if file_date <= date:
                valid_files.append((f, file_date))

        # Find the most recent file from the valid ones
        if valid_files:
            latest_file = max(valid_files, key=lambda x: x[1])
            with open(latest_file[0], 'rb') as f:
                return pickle.load(f)
        return {}

    def save_cached_features(self, dk, cached_features, date):
        cache_file = self.get_cache_filename(dk, date)
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_features, f)

    def should_use_cached_features(self, cached_features, label, last_candle_date):
        return label not in cached_features or \
               last_candle_date - pd.to_datetime(cached_features[label]['date']) >= timedelta(days=self.FEATURE_SELECT_PERIOD_DAYS)

    def feature_selection_process(self, data_dictionary, label, dk: FreqaiDataKitchen):
        # Initialize WandB
        self.wandb_init(name=f"{self.MODEL_IDENTIFIER}_{dk.pair}.feature_select[{label}]",
                        job_type="feature_select",
                        config={
                            "SELECT_FEATURES_ITERATIONS": self.SELECT_FEATURES_ITERATIONS,
                            "NUM_FEATURES_TO_SELECT": self.NUM_FEATURES_TO_SELECT,
                            "SELECT_FEATURES_STEPS": self.SELECT_FEATURES_STEPS,
                            "AUTODETECT_NUM_FEATURES_TO_SELECT": self.AUTODETECT_NUM_FEATURES_TO_SELECT,
                            "FEATURE_SELECT_LABEL": self.FEATURE_SELECT_LABEL,
                            "MODEL_IDENTIFIER": self.MODEL_IDENTIFIER,
                        })

        # Transform and prepare data
        x_train = data_dictionary["train_features"].copy().rename(columns=lambda x: x.replace('-', ':'))
        y_train = data_dictionary["train_labels"][label].copy()

        # Prepare eval data
        test_data = None
        if self.config["freqai"].get('data_split_parameters', {}).get('test_size', 0.05) > 0:
            x_eval = data_dictionary["test_features"].copy().rename(columns=lambda x: x.replace('-', ':'))
            y_eval = data_dictionary["test_labels"][label].copy()

            test_data = Pool(
                data=x_eval,
                label=y_eval
            )

        # Initialize and run the feature selection model
        feature_select_model = CatBoostClassifier(
            iterations=self.SELECT_FEATURES_ITERATIONS,
            **self.FEATURE_SELECT_MODEL_CONFIG
        )

        best_f = feature_select_model.select_features(
            x_train,
            y_train,
            eval_set=test_data,
            features_for_select=list(x_train.columns),
            num_features_to_select=self.NUM_FEATURES_TO_SELECT,
            steps=self.SELECT_FEATURES_STEPS,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
            shap_calc_type=EShapCalcType.Exact,
            train_final_model=False,
            logging_level=self.config["freqai"]["model_training_parameters"].get("logging_level", "Silent"),
            plot=False,
        )

        # Analyze results
        loss_graph = pd.DataFrame({"loss_values": best_f['loss_graph']['loss_values'],
                                "removed_features_count": best_f['loss_graph']['removed_features_count']})
        optimal_features = x_train.shape[1] - loss_graph[loss_graph['loss_values'] == loss_graph['loss_values'].min()]['removed_features_count'].max()
        min_loss_value = loss_graph["loss_values"].min()
        # last loss value
        final_loss_value = loss_graph["loss_values"].iloc[-1]

        self.log_to_wandb(loss_graph, best_f, optimal_features, min_loss_value, final_loss_value)
        wandb.finish()

        log(f'Now optimal number of features = {optimal_features} with value = {min_loss_value}')

        return best_f, optimal_features, min_loss_value

    def log_to_wandb(self, loss_graph, best_f, optimal_features, min_loss_value, final_loss_value):
        # Log the table to wandb
        wandb.log({"Loss Graph": wandb.Table(dataframe=loss_graph)})
        wandb.log({"Optimal Features": optimal_features, "Minimum Loss Value": min_loss_value, "Final Loss Value": final_loss_value})

        # Convert selected and eliminated features to wandb.Table
        # selected_features_table = wandb.Table(data=[best_f['selected_features_names']], columns=["Selected Features"])
        # eliminated_features_table = wandb.Table(data=[best_f['eliminated_features_names']], columns=["Eliminated Features"])

        # Log the tables to wandb
        # wandb.log({"Selected Features": best_f['selected_features_names'], "Eliminated Features": eliminated_features_table})

    def feature_select(self, data_dictionary, dk: FreqaiDataKitchen):
        selected_features_all_labels = {}
        last_candle_date = pd.to_datetime(data_dictionary["train_dates"].iloc[-1])

        cached_features = self.load_cached_features(dk, last_candle_date)

        for label in data_dictionary["train_labels"].columns:
            if not self.should_use_cached_features(cached_features, label, last_candle_date):
                selected_features_all_labels[label] = cached_features[label]['selected_features']
                continue

            log(f'Selecting features for label = {label}')
            best_f, optimal_features, min_loss_value = self.feature_selection_process(data_dictionary, label, dk=dk)

            # Save the selected features and last run date
            # TODO: when it creates a file for a new date, it will save also features from previous runs with old date
            # in ideal scenario it should not affect the system, but it's better to fix it someday
            selected_features_all_labels[label] = [x.replace(':', '-') for x in best_f['selected_features_names']]
            cached_features[label] = {
                'date': last_candle_date,
                'selected_features': selected_features_all_labels[label],
                'eliminated_features': [x.replace(':', '-') for x in best_f['eliminated_features_names']],
            }

            # Save the updated cached features
            self.save_cached_features(dk, cached_features, last_candle_date)

        return selected_features_all_labels

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        log(f'.predict with model = {self.MODEL_IDENTIFIER}')

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = filtered_df

        selected_features_all_labels = dk.data['selected_features']

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        predictions = self.model.predict(dk.data_dictionary["prediction_features"], selected_features_all_labels)
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))

        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"], selected_features_all_labels)
        if self.CONV_WIDTH == 1:
            predictions_prob = np.reshape(predictions_prob, (-1, len(self.model.classes_)))
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        pred = pred_df.tail(1).squeeze().to_dict()
        log(f"predictions = maxima={pred['maxima']}, minima={pred['minima']}, trend_long={pred['trend_long']}, trend_short={pred['trend_short']}")

        return (pred_df, dk.do_predict)

    def define_data_pipeline(self, threads) -> Pipeline:
        """
        User defines their custom feature pipeline here (if they wish)
        """
        feature_pipeline = Pipeline([
            ('scaler', SKLearnWrapper(RobustScaler())),
            ('di', ds.DissimilarityIndex(di_threshold=10, n_jobs=threads))
        ])

        return feature_pipeline

    def define_label_pipeline(self, threads) -> Pipeline:
        """
        User defines their custom label pipeline here (if they wish)
        """
        label_pipeline = Pipeline([
            ('scaler', SKLearnWrapper(RobustScaler())),
        ])

        return label_pipeline