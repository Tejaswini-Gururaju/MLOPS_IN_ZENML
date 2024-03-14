import numpy as np
import pandas as pd
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from pipelines.utils import get_data_for_test
import json
from steps.data_cleaning import clean_data
from steps.data_ingestion import ingest_data
from steps.model_evaluation import evaluate_model
from steps.model_train import train_model

docker_settings=DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    minimum_accuracy = 0.92

@step(enable_cache=False)
def dynamic_importer() -> str:
    data=get_data_for_test()
    return data

@step
def deployment_trigger(accuracy:float,config: DeploymentTriggerConfig):
    return accuracy > config.minimum_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name :str
    step_name:str
    running :bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name:str,
    pipeline_step_name:str,
    running:bool = True,
    model_name: str ="model",
) -> MLFlowDeploymentService:
    
    mlflow_model_deployer_component=MLFlowModelDeployer.get_active_model_deployer()
    existing_services=mlflow_model_deployer_component.find_model_server(


        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment services found for pipeline {pipeline_name}"
            f"step {pipeline_step_name} and model {model_name}"
            f"pipeline for the {model_name} model is currently running"
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray :
    
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False,settings={"docker" : docker_settings})
def continious_deployment_pipeline(data_path:str,min_accuracy:float=0.92, workers:int=1,timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT,):
    df=ingest_data(data_path=data_path)
    x_train,x_test,y_train,y_test=clean_data(df)
    model = train_model(x_train,x_test,y_train,y_test)
    r2_score,rmse=evaluate_model(model,x_test,y_test)
    deployment_decision=deployment_trigger(r2_score)
    mlflow_model_deployer_step(model=model,deploy_decision=deployment_decision,workers=workers,timeout=timeout)

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)


