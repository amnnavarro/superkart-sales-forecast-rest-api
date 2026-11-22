# SuperKart Streamlit app (frontend)
import io
import streamlit as st
import pandas as pd
import requests

# backend address on the Docker network
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Predict the total sales of a product in a store, one record at a time or in batch.")

# online prediction
st.subheader("Online Prediction")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Product**")
    product_id_char = st.selectbox("Product ID prefix", ["FD", "DR", "NC"])
    product_type = st.selectbox("Product type", [
        "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
        "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
        "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"], index=5)
    product_sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.66, step=0.1)
    product_area = st.number_input("Allocated display area (ratio)", min_value=0.0, max_value=1.0,
                                   value=0.027, step=0.001, format="%.3f")
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, step=1.0)
with col2:
    st.markdown("**Store**")
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
    store_city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1",
                                             "Supermarket Type2", "Food Mart"], index=2)
    store_year = st.number_input("Store establishment year", min_value=1950, max_value=2030, value=2009, step=1)

# payload with the model features
payload = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": product_area,
    "Product_MRP": product_mrp,
    "Product_Type": product_type,
    "Store_Establishment_Year": store_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        if response.status_code == 200:
            st.success(f"Predicted Sales: {response.json()['Predicted Sales']:,.2f}")
        else:
            st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# batch prediction
st.subheader("Batch Prediction")

# model columns (older files are also accepted)
REQUIRED_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Product_Type", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Id_char",
]
CORE_COLUMNS = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
                "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char"]
st.caption("Full schema: " + ", ".join(REQUIRED_COLUMNS) +
           ". Files that have Store_Age_Years instead of Store_Establishment_Year, or no Product_Type, "
           "are also accepted (a default is used and you are told which).")

# sample file to show the format
sample_df = pd.DataFrame([
    [12.66, "Low Sugar", 0.027, 117.08, "Frozen Foods", 2009, "Medium", "Tier 2", "Supermarket Type2", "FD"],
    [16.20, "Regular", 0.048, 182.50, "Dairy", 1987, "High", "Tier 2", "Supermarket Type1", "FD"],
    [9.80, "No Sugar", 0.062, 95.40, "Household", 1998, "Small", "Tier 3", "Food Mart", "NC"],
    [13.50, "Low Sugar", 0.031, 150.90, "Snack Foods", 1999, "Medium", "Tier 1", "Departmental Store", "FD"],
], columns=REQUIRED_COLUMNS)
st.download_button("Download sample CSV", sample_df.to_csv(index=False),
                   file_name="superkart_sample_batch.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload a CSV file with the model features", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
    cols = set(uploaded_df.columns)

    # check the columns before calling the API
    missing = [c for c in CORE_COLUMNS if c not in cols and not (c == "Product_Id_char" and "Product_Id" in cols)]
    if "Store_Establishment_Year" not in cols and "Store_Age_Years" not in cols:
        missing.append("Store_Establishment_Year (or Store_Age_Years)")

    if missing:
        st.error(f"The file is missing required columns: {missing}. Download the sample CSV above to see the format.")
    else:
        if "Product_Type" not in cols:
            st.info("No Product_Type column: the most frequent product type from training will be used.")
        if "Store_Establishment_Year" not in cols:
            st.info("Store_Establishment_Year will be computed from Store_Age_Years.")

        if st.button("Predict Batch", type="primary"):
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                         files={"file": file_bytes}, timeout=120)
                if response.status_code == 200:
                    # show the results as a table with a download
                    result = pd.DataFrame(list(response.json().items()), columns=["Row", "Predicted Sales"])
                    st.success("Batch predictions completed!")
                    if response.headers.get("X-Notes"):
                        st.warning(response.headers["X-Notes"])
                    st.dataframe(result)
                    st.download_button("Download predictions (CSV)", result.to_csv(index=False),
                                       file_name="superkart_predictions.csv", mime="text/csv")
                else:
                    st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to the prediction API: {e}")
