# =========================================================
# TITANIC SURVIVAL PREDICTION SYSTEM
# Streamlit + TensorFlow Deployment
# =========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model("titanic_ann_model.h5")

# =========================================================
# HEADER SECTION
# =========================================================

st.markdown("""
# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction
""")

st.image(
    "https://images.unsplash.com/photo-1529429617124-aee711a5ac1c",
    use_container_width=True
)

# =========================================================
# PROJECT DESCRIPTION
# =========================================================

with st.container():

    st.markdown("""
    ## 📌 Project Description
    
    This AI-powered application predicts whether a passenger
    would survive during the Titanic disaster using:
    
    - Artificial Neural Networks (ANN)
    - TensorFlow Deep Learning Model
    - Passenger Information
    
    The system uses:
    
    - Passenger Class
    - Age
    - Fare
    
    to estimate survival probability.
    """)

# =========================================================
# INPUT AREA
# =========================================================

st.markdown("## 🧾 Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

# =========================================================
# NORMALIZATION
# =========================================================

# Dummy dataset for scaling consistency
dummy_data = np.array([
    [1, 1, 0],
    [3, 80, 600]
])

scaler = MinMaxScaler()

scaler.fit(dummy_data)

input_data = np.array([[pclass, age, fare]])

input_scaled = scaler.transform(input_data)

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Survival"):

    prediction = model.predict(input_scaled)

    probability = float(prediction[0][0])

    # =====================================================
    # PREDICTION LOGIC
    # =====================================================

    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence = probability * 100

    # =====================================================
    # OUTPUT AREA
    # =====================================================

    st.markdown("## 🎯 Prediction Result")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            label="Prediction",
            value=result
        )

    with col5:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.2f}"
        )

    with col6:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    st.markdown("## 📊 Probability Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [survive_prob, nonsurvive_prob]
    })

    fig, ax = plt.subplots()

    ax.bar(
        chart_data["Category"],
        chart_data["Probability"]
    )

    ax.set_ylabel("Probability")

    st.pyplot(fig)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
---
### Developed using:
- Streamlit
- TensorFlow
- Python
- Deep Learning
""")