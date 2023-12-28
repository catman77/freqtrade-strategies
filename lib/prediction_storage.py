from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql import func

class PredictionStorage:
    # Define Base at the class level
    Base = declarative_base()

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = self.create_db_engine()
        self.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Define the table structure
        metadata = MetaData()
        self.tm_predictions_table = Table('tm_predictions', metadata, autoload_with=self.engine)

    def create_db_engine(self):
        engine = create_engine(self.connection_string, pool_size=10, max_overflow=20)

        return engine

    def save_prediction(self,
                        model_version: str,
                        pair: str,
                        timeframe: str,
                        trend_long: float,
                        trend_short: float,
                        maxima: float,
                        minima: float,
                        trend_long_roc_auc: float,
                        trend_long_f1: float,
                        trend_long_logloss: float,
                        trend_long_accuracy: float,
                        trend_short_roc_auc: float,
                        trend_short_f1: float,
                        trend_short_logloss: float,
                        trend_short_accuracy: float,
                        extrema_maxima_roc_auc: float,
                        extrema_maxima_f1: float,
                        extrema_maxima_logloss: float,
                        extrema_maxima_accuracy: float,
                        extrema_minima_roc_auc: float,
                        extrema_minima_f1: float,
                        extrema_minima_logloss: float,
                        extrema_minima_accuracy: float,
                        candle_time: datetime):
        """
        Save a prediction record to the database.

        Parameters:
        - model_version (str): Version of the predictive model.
        - pair (str): Currency pair or trading pair.
        - timeframe (str): Timeframe of the prediction.
        - trend_long (float): Probability of the long trend.
        - trend_short (float): Probability of the short trend.
        - maxima (float): Indicates if a maxima extremum is reached (1.0 for true, 0.0 for false).
        - minima (float): Indicates if a minima extremum is reached (1.0 for true, 0.0 for false).
        - trend_long_roc_auc (float): ROC AUC metric for the long trend prediction.
        - trend_long_f1 (float): F1 score for the long trend prediction.
        - trend_long_logloss (float): Logarithmic loss for the long trend prediction.
        - trend_long_accuracy (float): Accuracy metric for the long trend prediction.
        - trend_short_roc_auc (float): ROC AUC metric for the short trend prediction.
        - trend_short_f1 (float): F1 score for the short trend prediction.
        - trend_short_logloss (float): Logarithmic loss for the short trend prediction.
        - trend_short_accuracy (float): Accuracy metric for the short trend prediction.
        - extrema_maxima_roc_auc (float): ROC AUC metric for predicting maxima extremum.
        - extrema_maxima_f1 (float): F1 score for predicting maxima extremum.
        - extrema_maxima_logloss (float): Logarithmic loss for predicting maxima extremum.
        - extrema_maxima_accuracy (float): Accuracy metric for predicting maxima extremum.
        - extrema_minima_roc_auc (float): ROC AUC metric for predicting minima extremum.
        - extrema_minima_f1 (float): F1 score for predicting minima extremum.
        - extrema_minima_logloss (float): Logarithmic loss for predicting minima extremum.
        - extrema_minima_accuracy (float): Accuracy metric for predicting minima extremum.
        - candle_time (datetime): The timestamp of the prediction.
        """
        session = self.Session()
        new_prediction = TMPredictions(
            model_version=model_version,
            pair=pair,
            timeframe=timeframe,
            trend_long=trend_long,
            trend_short=trend_short,
            maxima=maxima,
            minima=minima,
            trend_long_roc_auc=trend_long_roc_auc,
            trend_long_f1=trend_long_f1,
            trend_long_logloss=trend_long_logloss,
            trend_long_accuracy=trend_long_accuracy,
            trend_short_roc_auc=trend_short_roc_auc,
            trend_short_f1=trend_short_f1,
            trend_short_logloss=trend_short_logloss,
            trend_short_accuracy=trend_short_accuracy,
            extrema_maxima_roc_auc=extrema_maxima_roc_auc,
            extrema_maxima_f1=extrema_maxima_f1,
            extrema_maxima_logloss=extrema_maxima_logloss,
            extrema_maxima_accuracy=extrema_maxima_accuracy,
            extrema_minima_roc_auc=extrema_minima_roc_auc,
            extrema_minima_f1=extrema_minima_f1,
            extrema_minima_logloss=extrema_minima_logloss,
            extrema_minima_accuracy=extrema_minima_accuracy,
            candle_time=candle_time
        )
        session.add(new_prediction)
        session.commit()
        session.close()

class TMPredictions(PredictionStorage.Base):
    __tablename__ = 'tm_predictions'
    id = Column(Integer, primary_key=True)
    model_version = Column(String)
    pair = Column(String)
    timeframe = Column(String)

    # prediction fields
    trend_long = Column(Float)
    trend_short = Column(Float)
    maxima = Column(Float)
    minima = Column(Float)

    # Trend Long metrics
    trend_long_roc_auc = Column(Float)
    trend_long_f1 = Column(Float)
    trend_long_logloss = Column(Float)
    trend_long_accuracy = Column(Float)

    # Trend Short metrics
    trend_short_roc_auc = Column(Float)
    trend_short_f1 = Column(Float)
    trend_short_logloss = Column(Float)
    trend_short_accuracy = Column(Float)

    # Extrema Maxima metrics
    extrema_maxima_roc_auc = Column(Float)
    extrema_maxima_f1 = Column(Float)
    extrema_maxima_logloss = Column(Float)
    extrema_maxima_accuracy = Column(Float)

    # Extrema Minima metrics
    extrema_minima_roc_auc = Column(Float)
    extrema_minima_f1 = Column(Float)
    extrema_minima_logloss = Column(Float)
    extrema_minima_accuracy = Column(Float)

    # Other fields
    created_at = Column(DateTime(timezone=True), default=func.now())
    candle_time = Column(DateTime(timezone=True))