import pandas as pd
import numpy as np
import talib
import pandas_ta as pta
from abc import ABC, abstractmethod

class SuperFeature(ABC):
    @abstractmethod
    def indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values and return them as a DataFrame."""
        pass

    @abstractmethod
    def custom_features(self, ohlcv_df: pd.DataFrame, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from indicator values and return them as a DataFrame."""
        return None

    @abstractmethod
    def signals(self, ohlcv_df: pd.DataFrame, indicators_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators and features, return as a DataFrame."""
        pass

    def crossed_up(self, series1: pd.Series, series2) -> pd.Series:
        if isinstance(series2, pd.Series):
            return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
        elif isinstance(series2, (int, float)):
            return (series1 > series2) & (series1.shift(1) <= series2)
        else:
            raise ValueError("Unsupported type for series2")

    def crossed_down(self, series1: pd.Series, series2) -> pd.Series:
        if isinstance(series2, pd.Series):
            return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
        elif isinstance(series2, (int, float)):
            return (series1 < series2) & (series1.shift(1) >= series2)
        else:
            raise ValueError("Unsupported type for series2")

    def hlc3(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        return (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3


    def standard_features(self, ohlcv_df: pd.DataFrame, indicators_df: pd.DataFrame) -> pd.DataFrame:
        return None


    def calculate(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators, features, and signals, and concatenate the results."""
        indicators_df = self.indicators(ohlcv_df)
        standard_features_df = self.standard_features(ohlcv_df, indicators_df)
        features_df = self.features(ohlcv_df, indicators_df)
        signals_df = self.signals(ohlcv_df, indicators_df, features_df)

        # Concatenate the results
        result_df = pd.concat([indicators_df, standard_features_df, features_df, signals_df], axis=1)

        return result_df
