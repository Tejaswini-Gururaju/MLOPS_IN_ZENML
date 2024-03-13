import logging
from sklearn.linear_model import LinearRegression
from abc import abstractmethod,ABC

class Model(ABC):


    @abstractmethod
    def train(self,x_train,y_train):


        pass
class LinearRegressionModel(Model):

    def train(self, x_train, y_train,**kwargs):
        try:

            reg = LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("Model training completed")
            return reg
        
        except Exception as e:
            logging.info(f"Model training failed {e}")
            raise e


        





    

