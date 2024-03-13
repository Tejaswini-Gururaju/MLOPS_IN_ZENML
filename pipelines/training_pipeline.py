from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_cleaning import clean_data
from steps.model_evaluation import evaluate_model
from steps.model_train import train_model
# from pipelines.training_pipeline import train_pipeline
from src.cleaning_data import DataCleaning,DataDivideStrategy,DataPreprocessingStrategy
from src.model_development import LinearRegressionModel
from src.evaluation_model import MSE,R2,RMSE
from pathlib import Path


@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df = ingest_data(data_path)
    x_train,x_test,y_train,y_test=clean_data(df)
    model = train_model(x_train,x_test,y_train,y_test)
    r2_score,rmse=evaluate_model(model,x_test,y_test)
    return r2_score, rmse

