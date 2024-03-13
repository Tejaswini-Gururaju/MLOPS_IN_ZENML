import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining startegy for handling data
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreprocessingStrategy(DataStrategy):
    '''
    strategy for preprocessing data

    '''
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        preprocessing data
        '''
        try:
            data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ],axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(),inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(),inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(),inplace=True)
            data["review_comment_message"].fillna("No review",inplace=True)

            data=data.select_dtypes(include=[np.number])
            cols_to_drop= ["customer_zip_code_prefix","order_item_id"]
            data=data.drop(cols_to_drop,axis=1)

            return data
        
        except Exception as e:
            logging.info(f"Error in preprocessing data {e}")
            raise e
            

class DataDivideStrategy(DataStrategy):
    """
    strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop(["review_score"],axis=1)
            y = data["review_score"]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            return x_train,x_test,y_train,y_test
        
        except Exception as e:
            logging.info(f"Error is splitting the data {e}")
            raise e
        
class DataCleaning:
    """
    Class which processes the data and divides the data for training and testing
    """

    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        
        except  Exception as e:
            logging.info(f"Error in handling data {e}")
            raise e
        

        

