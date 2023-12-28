from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql import func
import logging

logger = logging.getLogger(__name__)

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

    def save_prediction(self, model_version: str, pair: str, timeframe: str,
                        trend_long: float, trend_short: float,
                        maxima: float, minima: float,
                        trend_long_roc_auc: float, trend_long_f1: float,
                        trend_long_logloss: float, trend_long_accuracy: float,
                        trend_short_roc_auc: float, trend_short_f1: float,
                        trend_short_logloss: float, trend_short_accuracy: float,
                        extrema_maxima_roc_auc: float, extrema_maxima_f1: float,
                        extrema_maxima_logloss: float, extrema_maxima_accuracy: float,
                        extrema_minima_roc_auc: float, extrema_minima_f1: float,
                        extrema_minima_logloss: float, extrema_minima_accuracy: float,
                        candle_time: datetime):
        """
        Save a prediction record to the database.
        """
        MAX_RETRIES = 3
        attempts = 0

        while attempts < MAX_RETRIES:
            session = self.Session()
            try:
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
                return  # Exit the method if save is successful
            except Exception as e:
                session.rollback()
                attempts += 1
                logger.error(f"Attempt {attempts} failed: {e}")
                if attempts == MAX_RETRIES:
                    logger.error("Maximum retry attempts reached. Aborting save.")
                    # Optionally re-raise the exception
                    # raise
            finally:
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