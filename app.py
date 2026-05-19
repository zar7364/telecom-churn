import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    layout="wide"
)

# =========================
# LOAD LOGO
# =========================

logo = Image.open("logo.png")

# =========================
# HEADER
# =========================

col1, col2 = st.columns([1.5, 5])

with col1:

    st.image(
        logo,
        width=180
    )

with col2:

    st.title(
        "Telecom Customer Churn Prediction"
    )

    st.write(
        """
        ### Laporan Tugas Akhir
        
        Sains & Analisis Data
        
        **Nezar Abdilah Prakasa**  
        563414
        
        Program Magister Kecerdasan Artifisial  
        Universitas Gadjah Mada
        """
    )

st.divider()

# =========================
# DATA SOURCE
# =========================

st.subheader(
    "Data Source for Modeling"
)

st.write(
    """
    Dataset digunakan untuk proses modeling customer churn prediction 
    yang diperoleh dari Maven Analytics Data Playground.
    
    Link Dataset:
    https://mavenanalytics.io/data-playground/telecom-customer-churn
    """
)

st.info(
    """
    Upload data customer sendiri (.xlsx) dengan format kolom 
    yang sesuai seperti dataset asli pada link Maven Analytics.
    """
)

# =========================
# LOAD MODEL FILES
# =========================

model = pickle.load(
    open(
        'xgboost_churn_model.pkl',
        'rb'
    )
)

scaler = pickle.load(
    open(
        'scaler.pkl',
        'rb'
    )
)

selected_features = pickle.load(
    open(
        'selected_features.pkl',
        'rb'
    )
)

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=['xlsx']
)

# =========================
# SAMPLE DATA BUTTON
# =========================

use_sample = st.button(
    "Use Original Dataset"
)

# =========================
# LOAD DATA
# =========================

df = None

if uploaded_file is not None:

    df = pd.read_excel(
        uploaded_file
    )

elif use_sample:

    df = pd.read_excel(
        'telecom_customer_churn.xlsx'
    )

# =========================
# PROCESS DATA
# =========================

if df is not None:

    st.subheader(
        "Preview Data"
    )

    st.dataframe(
        df.head()
    )

    # =========================
    # REMOVE TARGET COLUMN
    # =========================

    if 'Customer Status' in df.columns:

        df = df.drop(
            columns=['Customer Status']
        )

    # =========================
    # KEEP SELECTED FEATURES
    # =========================

    available_features = [

        col for col in selected_features

        if col in df.columns
    ]

    X = df[
        available_features
    ].copy()

    # =========================
    # ADD MISSING FEATURES
    # =========================

    for feature in selected_features:

        if feature not in X.columns:

            X[feature] = 0

    # =========================
    # REORDER FEATURES
    # =========================

    X = X[
        selected_features
    ]

    # =========================
    # LABEL ENCODING
    # =========================

    mapping_dict = {

        'Internet Type': {

            'Cable': 0,
            'DSL': 1,
            'Fiber Optic': 2
        },

        'Online Security': {

            'No': 0,
            'Yes': 1
        },

        'Premium Tech Support': {

            'No': 0,
            'Yes': 1
        },

        'Contract': {

            'Month-to-Month': 0,
            'One Year': 1,
            'Two Year': 2
        },

        'Payment Method': {

            'Bank Withdrawal': 0,
            'Credit Card': 1,
            'Mailed Check': 2
        }
    }

    # =========================
    # APPLY ENCODING
    # =========================

    for col, mapping in mapping_dict.items():

        if col in X.columns:

            X[col] = X[col].map(
                mapping
            )

    # =========================
    # HANDLE NULL VALUES
    # =========================

    X = X.fillna(0)

    # =========================
    # SCALING
    # =========================

    X_scaled = scaler.transform(
        X
    )

    # =========================
    # PREDICTION
    # =========================

    predictions = model.predict(
        X_scaled
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:,1]

    # =========================
    # RESULT DATAFRAME
    # =========================

    result_df = df.copy()

    result_df['Prediction'] = predictions

    result_df['Prediction Label'] = (

        result_df['Prediction']

        .map({

            1: 'Churn',
            0: 'Stay'
        })
    )

    result_df['Churn Probability (%)'] = (

        probabilities * 100

    ).round(2)

    # =========================
    # SHOW RESULT
    # =========================

    st.subheader(
        "Prediction Result"
    )

    st.dataframe(
        result_df
    )

    # =========================
    # SUMMARY METRIC
    # =========================

    churn_count = (

        result_df['Prediction Label']

        == 'Churn'

    ).sum()

    stay_count = (

        result_df['Prediction Label']

        == 'Stay'

    ).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "Churn Customer",
        churn_count
    )

    col2.metric(
        "Stay Customer",
        stay_count
    )

    # =========================
    # DOWNLOAD RESULT
    # =========================

    csv = result_df.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(

        label="Download Prediction Result",

        data=csv,

        file_name='churn_prediction_result.csv',

        mime='text/csv'
    )