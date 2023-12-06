from datetime import datetime
import json
import logging
import sys
from pathlib import Path
import time
from typing import Any, Dict

from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, EShapCalcType, Pool

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import numpy as np
import numpy.typing as npt
import pandas as pd
import psutil
from pandas import DataFrame
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CatboostFeatureSelectedRegressorV1(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

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

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        # select features
        selected_features = self.feature_select(data_dictionary)
        dk.data['selected_features'] = selected_features

        # transform and prepare data
        train_data = Pool(
            data=data_dictionary["train_features"][selected_features],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            test_data = None
        else:
            test_data = Pool(
                data=data_dictionary["test_features"][selected_features],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
            )

        # train final model
        final_model = CatBoostRegressor(
            allow_writing_files=True,
            train_dir=Path(dk.data_path),
            random_seed=42,
            **self.model_training_parameters,
        )

        final_model.fit(X=train_data, eval_set=test_data,
                  log_cout=sys.stdout, log_cerr=sys.stderr)

        return final_model

    def feature_select(self, data_dictionary):
        # transform and prepare data
        x_train = data_dictionary["train_features"].copy()
        y_train = data_dictionary["train_labels"].copy()
        x_train = x_train.rename(columns=lambda x: x.replace('-', ':'))

        # prepare eval data
        test_data = None

        if self.config["freqai"].get('data_split_parameters', {}).get('test_size', 0.05) > 0:
            # replace '-' with ':' in test labels
            x_eval = data_dictionary["test_features"].copy().rename(columns=lambda x: x.replace('-', ':'))
            y_eval = data_dictionary["test_labels"].copy()

            test_data = Pool(
                data=x_eval,
                label=y_eval
            )


        # create model & select features
        feature_select_model = CatBoostRegressor(iterations=self.SELECT_FEATURES_ITERATIONS,
                                task_type="CPU",
                                eval_metric='RMSE',
                                colsample_bylevel=0.3,
                                depth=6,
                                boosting_type="Plain",
                                bootstrap_type="MVS",
                                l2_leaf_reg=9.0,
                                learning_rate=0.05,
                                save_snapshot=False,
                                allow_writing_files=False,
                                random_seed=42)

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

        try:
            loss_graph = pd.DataFrame({"loss_values": best_f['loss_graph']['loss_values'], "removed_features_count": best_f['loss_graph']['removed_features_count']})
            optimal_features = x_train.shape[1] - loss_graph[loss_graph['loss_values'] == loss_graph['loss_values'].min()]['removed_features_count'].max()
            logger.info(f'Now optimal number of features = {optimal_features}')
        except:
            pass

        # save best_f to json file
        with open(f'user_data/artifacts/{str(self.config["freqai"].get("identifier"))}_best_features_{datetime.now()}.json', 'w') as f:
            json.dump(best_f, f)

        new_best_features = [x.replace(':', '-') for x in best_f['selected_features_names']]
        return new_best_features


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

        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_df = dk.normalize_data_from_metadata(filtered_df)
        dk.data_dictionary["prediction_features"] = filtered_df

        selected_feature_names = dk.data['selected_features']
        selected_features = dk.data_dictionary["prediction_features"][selected_feature_names]

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk)

        start = time.time()
        predictions = self.model.predict(selected_features)
        time_spent = (time.time() - start)
        self.dd.update_metric_tracker('predict_time', time_spent, dk.pair)

        pred_df = DataFrame(predictions, columns=dk.label_list)

        print("Predictions: ", pred_df)
        pred_df = dk.denormalize_labels_from_metadata(pred_df)

        return (pred_df, dk.do_predict)