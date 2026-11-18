# ---------------------------------------------------------------
# SuperKart Sales Forecast – Streamlit UI (frontend)
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
import requests

# Backend reached through the Docker network (container name = "backend")
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.write("Predict the total sales of a product in a store, one record at a time or in batch.")

# ============================ ONLINE (single) ============================
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

# Build the JSON payload with exactly the features the model expects
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

# ============================ BATCH (CSV) ============================
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the model features", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                 files={"file": uploaded_file.getvalue()}, timeout=120)
        if response.status_code == 200:
            # API returns {row index: prediction}; show it as a table and offer a download
            result = pd.DataFrame(list(response.json().items()), columns=["Row", "Predicted Sales"])
            st.success("Batch predictions completed!")
            st.dataframe(result)
            st.download_button("Download predictions (CSV)", result.to_csv(index=False),
                               file_name="superkart_predictions.csv", mime="text/csv")
        else:
            st.error(f"API error ({response.status_code}): {response.json().get('error', response.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")
