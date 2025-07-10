from dagster import Definitions
from dagster import define_asset_job

from code_location_interview.code_location_interview.assets.magenta_interview.assets import (
    model_object,
    prod_data_raw,
    prod_data_preprocessed,
    prod_predictions_and_ab_test_split,
    precision_recall_analysis,  
    shap_top_predictions,
)

custom_execution_order_job = define_asset_job(
    name="custom_execution_order_job",
    selection=[
        "model_object",
        "prod_data_raw",
        "prod_data_preprocessed",
        "prod_predictions_and_ab_test_split",
        "precision_recall_analysis",
        "shap_top_predictions",
    ],
)


defs = Definitions(
    assets=[
        model_object,
        prod_data_raw,
        prod_data_preprocessed,
        prod_predictions_and_ab_test_split,
        precision_recall_analysis, 
        shap_top_predictions,
    ],
    jobs=[custom_execution_order_job]
)