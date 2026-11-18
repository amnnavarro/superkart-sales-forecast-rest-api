# ---------------------------------------------------------------
# SuperKart Sales Forecast – Flask REST API (backend)
# ---------------------------------------------------------------
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Create the Flask app (the Dockerfile/gunicorn refers to this object name)
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the serialized pipeline (preprocessing + tuned model) once at startup
model = joblib.load("superkart_model.joblib")

# Feature schema the model was trained on (order irrelevant, names must match)
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Product_Type", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Id_char",
]


def prepare_features(df):
    # Derive Product_Id_char if only Product_Id was sent, then select the model features
    df = df.copy()
    if "Product_Id_char" not in df.columns and "Product_Id" in df.columns:
        df["Product_Id_char"] = df["Product_Id"].str[:2]
    # Harmonize the one known label inconsistency
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURES]


# Home route (GET): quick check that the API is alive
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API!"


# Online inference (POST): one JSON record -> one prediction
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Request body must be a JSON object."}), 400

        input_df = prepare_features(pd.DataFrame([payload]))

        # float() converts the NumPy value so jsonify can serialize it
        prediction = round(float(model.predict(input_df)[0]), 2)
        return jsonify({"Predicted Sales": prediction})

    except ValueError as e:                       # bad / incomplete input
        return jsonify({"error": str(e)}), 400
    except Exception as e:                        # anything unexpected
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# Batch inference (POST): CSV file -> {row index: prediction}
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Upload a CSV file using the form key 'file'."}), 400

        input_df = pd.read_csv(request.files["file"])
        preds = model.predict(prepare_features(input_df))

        # Keys = row index of the uploaded file, values = predicted sales
        output = {str(i): round(float(p), 2) for i, p in zip(input_df.index, preds)}
        return jsonify(output)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


# Local run (in Docker, gunicorn starts the app instead)
if __name__ == "__main__":
    superkart_api.run(debug=True)
