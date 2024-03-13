import logging
import pandas as pd
from zenml import step
from src.cleaning_data import DataCleaning,DataDivideStrategy,DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"x_train"],
    Annotated[pd.DataFrame,"x_test"],
    Annotated[pd.Series ,"y_train"],
    Annotated[pd.Series ,"y_test"]

]:
    """
    Cleans the data and divides into train/test for modeltraining and evaluation

    Args :
         df : raw data
    Returns:
          training and testing data, training and testing labels
          
    """
    try:
        process_startegy=DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df,process_startegy)
        processed_data=data_cleaning.handle_data()

        divide_strategy=DataDivideStrategy()
        data_dividing=DataCleaning(processed_data,divide_strategy)
        x_train,x_test,y_train,y_test  = data_dividing.handle_data()
        logging.info(f"Data is divided for training and testing")
        return x_train,x_test,y_train,y_test


    except Exception as e:
        logging.info(f"Error in data cleaning {e}")
        raise e
    
