import logging
from abc import ABC,abstractmethod
import numpy as np
from numpy.core.multiarray import array as array
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt

class Evaluation(ABC):
    @abstractmethod

    def calculate_score(self,y_true:np.array,y_pred:np.array):
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that used mean squared error 
    """
    def calculate_score(self, y_true: np.array, y_pred: np.array):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info(f"MSE value is : {mse}")
            return mse
    
        except Exception as e:
            logging.info(f"Error in calculating the scores {e}")
            raise e
        
class R2(Evaluation):
    def calculate_score(self, y_true: np.array, y_pred: np.array):
        try:
            logging.info("Calculating R2 score")
            r2=r2_score(y_true,y_pred)
            logging.info(f"R2 score is : {r2}")
            return r2
    
        except Exception as e:
            logging.info(f"Error in calculating the scores {e}")
            raise e
        
class RMSE(Evaluation):
    def calculate_score(self, y_true: np.array, y_pred: np.array):
        try:
            logging.info("Calculating RMSE")
            rmse=sqrt(mean_squared_error(y_true,y_pred))
            logging.info(f"RMSE value is : {rmse}")
            return rmse
    
        except Exception as e:
            logging.info(f"Error in calculating the scores {e}")
            raise e
