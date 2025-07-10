import pandas as pd
import lightgbm as lgb
import pickle
from dagster import asset, Output, MetadataValue
import os
from sklearn.metrics import roc_auc_score,precision_score, recall_score, f1_score
import numpy as np
import shap
from sklearn.metrics import roc_auc_score



@asset
def prod_data_raw() -> pd.DataFrame:
    """Loads the out-of-sample (OOS) or production data from a fixed absolute path."""
    file_path = "/workspaces/data-scientist-at-magenta-/data/processed/X_oos.parquet"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    return pd.read_parquet(file_path, engine="pyarrow")


@asset
def model_object():
    """
    Loads the trained LightGBM model object from disk.
    This is needed to access feature names and use the model for inference.
    """
    with open("models/final_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@asset
def prod_data_preprocessed(prod_data_raw: pd.DataFrame, model_object) -> pd.DataFrame:
    """
    Preprocess the raw OOS/PROD data using the model's expected feature names.
    Casts selected columns to categorical and selects only model features.
    """
    df = prod_data_raw.copy()

    # Optional: convert these to categorical (if used in model)
    categorical_cols = [
        'has_special_offer',
        'is_magenta1_customer',
        'available_gb',
        'smartphone_brand',
        'has_multiple_contracts'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure column selection matches model training
    features_used = model_object.feature_name()
    df = df[[col for col in features_used if col in df.columns]]

    return df



@asset
def prod_predictions_and_ab_test_split(
    model_object, 
    prod_data_preprocessed: pd.DataFrame, 
    prod_data_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates probability and binary predictions using the loaded model.
    Attaches customer_id from the raw data for traceability.
    """
    # Get predicted probabilities (LightGBM Booster)
    probs = model_object.predict(prod_data_preprocessed)
    preds = (probs >= 0.43).astype(int)

    
    if "customer_id" in prod_data_raw.columns:
        customer_ids = prod_data_raw["customer_id"].reset_index(drop=True)
    else:
        customer_ids = pd.Series(range(len(probs)), name="customer_id")

    result = pd.DataFrame({
        "customer_id": customer_ids,
        "predicted_proba": probs,
        "predicted_label": preds
    })

    np.random.seed(42)  
    result["is_control"] = np.random.rand(len(result)) < 0.5

    os.makedirs("outputs", exist_ok=True)
    result.to_csv("outputs/prod_predictions.csv", index=False)

    return result



@asset
def prod_auc(prod_data_raw: pd.DataFrame, prod_predictions: pd.DataFrame) -> Output[float]:
    actuals = prod_data_raw["has_done_upselling"].reset_index(drop=True)
    probs = prod_predictions["predicted_proba"]

    auc_score = roc_auc_score(actuals, probs)

    # Save to local file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/prod_auc.txt", "w") as f:
        f.write(f"AUC Score: {auc_score:.4f}\n")

    # Cast to plain float for Dagster compatibility
    return Output(
        value=float(auc_score),
        metadata={"AUC Score": MetadataValue.float(float(auc_score))}
    )


@asset
def precision_recall_analysis(
    prod_predictions_and_ab_test_split: pd.DataFrame,
    prod_data_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates precision, recall, and F1 score across thresholds using OOS predictions.
    Saves thresholded scores and best threshold by F1.
    """
    y_true = prod_data_raw["has_done_upselling"].reset_index(drop=True)
    y_scores = prod_predictions_and_ab_test_split["predicted_proba"].reset_index(drop=True)

    # 10% step analysis
    display_thresholds = np.round(np.linspace(0.0, 1.0, 11), 2)
    display_scores = []

    for t in display_thresholds:
        y_pred_binary = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        display_scores.append((t, f1, precision, recall))

    score_df_display = pd.DataFrame(display_scores, columns=['threshold', 'f1', 'precision', 'recall'])

    # Fine-grained F1 optimization
    fine_thresholds = np.round(np.linspace(0.01, 1.00, 100), 2)
    fine_scores = []

    for t in fine_thresholds:
        y_pred_binary = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        fine_scores.append((t, f1))

    fine_score_df = pd.DataFrame(fine_scores, columns=['threshold', 'f1'])
    best_row = fine_score_df.loc[fine_score_df['f1'].idxmax()]

    # Logging results (for Dagster UI)
    print("Precision, Recall, F1 at thresholds (10% steps):")
    print(score_df_display.sort_values(by='threshold', ascending=False).to_string(index=False))
    print(f"\nMax F1 score: {best_row.f1:.3f} at threshold = {best_row.threshold:.2f}")

    # Save both for record
    os.makedirs("outputs", exist_ok=True)
    score_df_display.to_csv("outputs/precision_recall_display.csv", index=False)
    fine_score_df.to_csv("outputs/precision_recall_fine.csv", index=False)

    return score_df_display



@asset
def shap_top_predictions(
    model_object, 
    prod_data_preprocessed: pd.DataFrame, 
    prod_data_raw: pd.DataFrame
) -> str:
    """
    Generate SHAP values for top 10 predicted customers and export feature values + SHAP values to CSV.
    """
    # Get prediction scores
    probs = model_object.predict(prod_data_preprocessed)

    # Select top 10 indices with highest predicted probability
    top_indices = probs.argsort()[-10:][::-1]  # Top 10 highest
    top_customers = prod_data_preprocessed.iloc[top_indices].copy()

    # Run SHAP
    explainer = shap.TreeExplainer(model_object)
    shap_values = explainer.shap_values(prod_data_preprocessed)[1]  # Class 1 SHAP values

    # Prepare export table
    rows = []
    for i, idx in enumerate(top_indices):
        customer_id = prod_data_raw.iloc[idx].get("customer_id", f"row_{idx}")
        row_data = {
            "rank": i + 1,
            "customer_id": customer_id,
            "predicted_proba": probs[idx]
        }
        # Add feature values and SHAP values correctly
        for j, feat in enumerate(prod_data_preprocessed.columns):
            row_data[f"value__{feat}"] = prod_data_preprocessed.iloc[idx][feat]
            row_data[f"shap__{feat}"] = shap_values[idx, j] 

        rows.append(row_data)

    output_df = pd.DataFrame(rows)

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/shap_top_predictions.csv"
    output_df.to_csv(output_path, index=False)

    return output_path







