from dagster import Definitions
from code_location_interview.code_location_interview.assets.magenta_interview.assets import (
    model_object,
    prod_data_raw,
    prod_data_preprocessed,
    prod_predictions,
    precision_recall_analysis,  
    shap_top_predictions,
)

defs = Definitions(
    assets=[
        model_object,
        prod_data_raw,
        prod_data_preprocessed,
        prod_predictions,
        precision_recall_analysis, 
        shap_top_predictions,
    ]
)