from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func, exc
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class TM5Predictions(Base):
    __tablename__ = 'tm5_predictions'
    id = Column(Integer, primary_key=True)
    model_version = Column(String)
    pair = Column(String)
    timeframe = Column(String)
    extrema = Column(Float)
    minima_sort_threshold = Column(Float)
    maxima_sort_threshold = Column(Float)
    range_max = Column(Float)
    range_min = Column(Float)
    di_values = Column(Float)
    di_cutoff = Column(Float)
    created_at = Column(DateTime(timezone=True), default=func.now())
    candle_time = Column(DateTime(timezone=True))

class TM5PredictionStorage:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _retry_policy(func):
        """
        Decorator to implement retry policy for database operations.
        """
        def wrapper(self, *args, **kwargs):
            MAX_ATTEMPTS = 3
            attempts = 0
            while attempts < MAX_ATTEMPTS:
                try:
                    return func(self, *args, **kwargs)
                except exc.IntegrityError:
                    print("Integrity error, no retry needed.")
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    if attempts == MAX_ATTEMPTS:
                        print("Maximum retry attempts reached. Operation failed.")
        return wrapper

    @_retry_policy
    def add_prediction(self, **kwargs):
        session = self.Session()
        try:
            prediction = TM5Predictions(**kwargs)
            session.add(prediction)
            session.commit()
            print("New prediction added successfully.")
        finally:
            session.close()

    @_retry_policy
    def update_prediction(self, prediction_id, **kwargs):
        session = self.Session()
        try:
            session.query(TM5Predictions).filter_by(id=prediction_id).update(kwargs)
            session.commit()
            print("Prediction updated successfully.")
        finally:
            session.close()

    @_retry_policy
    def delete_prediction(self, prediction_id):
        session = self.Session()
        try:
            prediction_to_delete = session.query(TM5Predictions).filter_by(id=prediction_id).first()
            if prediction_to_delete:
                session.delete(prediction_to_delete)
                session.commit()
                print("Prediction deleted successfully.")
        finally:
            session.close()