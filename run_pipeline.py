from pipelines.training_pipeline import train_pipeline
from pathlib import Path

if __name__=="__main__":
    #Run the pipeline
    train_pipeline(data_path="D:\MLOPS_IN_ZENML\data\olist_customers_dataset.csv")