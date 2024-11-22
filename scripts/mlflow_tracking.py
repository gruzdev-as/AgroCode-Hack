import mlflow
import catboost
import pandas as pd

from typing import Dict

def log_model_info_metrics(
    model: catboost.core.CatBoostClassifier | catboost.core.CatBoostRegressor,
    metric_dict: Dict[str, float], 
    model_postfix: str,
    input_example: pd.DataFrame
) -> None: 
    """Log data using mlflow

    Args:
        model (catboost.core.CatBoostClassifier | catboost.core.CatBoostRegressor): A trained model
        metric_dict (_type_): Dict of metrics to log
        model_postfix (str): Postfix for saving model artifacts
        input_example (pd.core.DataFrame): Example of input for model
    """

    try:
        model_parameters = model.get_params()
        mlflow.log_params(model_parameters)

        for metric_name, metric_value in metric_dict.values():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.catboost.log_model(
            artifact_path=f"catboost_model_{model_postfix}.cbm", 
            cb_model=model,
            input_example=input_example)
    
    except Exception as e:
        print(f'Error: {e}')
    
    finally:
        print('Logged succesfully')

