import logging
import pandas as pd
from zenml import step
import mlflow
from zenml.client import Client
from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step
def train_model(x_train:pd.DataFrame,
                x_test:pd.DataFrame,
                y_train:pd.DataFrame,
                y_test:pd.DataFrame,
                config:ModelNameConfig) -> RegressorMixin:
    '''
    Trains the model on the ingested data

    Args: pd.DataFrame : ingested data
    '''
    try :
        model = None

        if config.model_name=="LinearRegression":
           mlflow.sklearn.autolog()

           model = LinearRegressionModel()
      
           trained_model=model.train(x_train,y_train)
           return trained_model
        else:
           raise ValueError(f"Model {config.model_name} is not supported")
        
    except Exception as e:
        logging.info(f"Error in training model {e} ")
        raise e
    
    






    

