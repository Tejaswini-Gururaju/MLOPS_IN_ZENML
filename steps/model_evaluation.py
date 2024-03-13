import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation_model import MSE,R2,RMSE
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step
def evaluate_model(model:RegressorMixin,
                   x_test: pd.DataFrame,
                   y_test:pd.DataFrame) -> Tuple[
                         Annotated[float,"r2_score"],
                         Annotated[float,"rmse"]
                   ]:
    
    try:
            prediction=model.predict(x_test)
            mse_class=MSE()
            mse= mse_class.calculate_score(y_test,prediction)
            mlflow.log_metric("mse",mse)
    
            prediction=model.predict(x_test)
            rmse_class=RMSE()
            rmse = rmse_class.calculate_score(y_test,prediction)
            mlflow.log_metric("rmse",rmse)


            prediction=model.predict(x_test)
            r2_class=R2()
            r2= r2_class.calculate_score(y_test,prediction)
            mlflow.log_metric("r2",r2)


            return r2, rmse
    except Exception as e:
          logging.info(f"Error in evaluating the model {e}")
          raise e




