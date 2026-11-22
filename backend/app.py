# SuperKart Flask API (backend)
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# create the app
superkart_api = Flask("SuperKart Sales Forecaster")

# load the model once at startup
model = joblib.load("superkart_model.joblib")

# features the model was trained on
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Product_Type", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Id_char",
]

# only used to turn Store_Age_Years into a year
REFERENCE_YEAR = 2025


def prepare_features(df):
    # returns the features and notes about any default used
    df = df.copy()
    notes = []

    # get Product_Id_char from Product_Id if needed
    if "Product_Id_char" not in df.columns and "Product_Id" in df.columns:
        df["Product_Id_char"] = df["Product_Id"].str[:2]

    # fix the 'reg' label
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # older files: Store_Age_Years to Store_Establishment_Year
    if "Store_Establishment_Year" not in df.columns and "Store_Age_Years" in df.columns:
        df["Store_Establishment_Year"] = REFERENCE_YEAR - df["Store_Age_Years"]
        notes.append(f"Store_Establishment_Year was computed as {REFERENCE_YEAR} - Store_Age_Years.")

    # Product_Type can't be rebuilt from Product_Type_Category, leave it empty,
    # the pipeline imputer fills it with the most common type
    if "Product_Type" not in df.columns:
        df["Product_Type"] = np.nan
        notes.append("Product_Type was not provided: the most frequent product type in the training data was used.")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURES], notes


# home route
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API!"


# single prediction
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be a JSON object."}), 400

        input_df, notes = prepare_features(pd.DataFrame([payload]))

        # float() so jsonify can handle it
        result = {"Predicted Sales": round(float(model.predict(input_df)[0]), 2)}
        if notes:
            result["Notes"] = notes
        return jsonify(result)

    except ValueError as e:                       # bad / incomplete input
        return jsonify({"error": str(e)}), 400
    except Exception as e:                        # anything unexpected
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# batch prediction
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Upload a CSV file using the form key 'file'."}), 400

        input_df = pd.read_csv(request.files["file"])
        features, notes = prepare_features(input_df)
        preds = model.predict(features)

        # row index: predicted sales
        response = jsonify({str(i): round(float(p), 2) for i, p in zip(input_df.index, preds)})
        if notes:
            response.headers["X-Notes"] = " | ".join(notes)   # the body stays {row: prediction}
        return response

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


# local run
if __name__ == "__main__":
    superkart_api.run(debug=True)
