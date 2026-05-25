import streamlit as st
import math
import json

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Titanic ANN Prediction",
    page_icon="🚢",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD WEIGHTS
# =====================================================

with open("ann_weights.json", "r") as f:
    weights = json.load(f)

# Extract weights
w1 = weights["w1"]
w2 = weights["w2"]
w3 = weights["w3"]

w4 = weights["w4"]
w5 = weights["w5"]
w6 = weights["w6"]

w7 = weights["w7"]
w8 = weights["w8"]

bh1 = weights["bh1"]
bh2 = weights["bh2"]

bo = weights["bo"]

# =====================================================
# HEADER
# =====================================================

st.markdown("""
# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction
""")

st.write("---")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Passenger Information")

x1 = st.sidebar.slider(
    "Passenger Class (Normalized)",
    0.0,
    1.0,
    0.2
)

x2 = st.sidebar.slider(
    "Age (Normalized)",
    0.0,
    1.0,
    0.24
)

x3 = st.sidebar.slider(
    "Fare (Normalized)",
    0.0,
    1.0,
    0.80
)

# =====================================================
# SIGMOID FUNCTION
# =====================================================

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button("Predict Survival")

if predict:

    # =================================================
    # FORWARD PROPAGATION
    # =================================================

    zh1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + bh1
    zh2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + bh2

    h1 = sigmoid(zh1)
    h2 = sigmoid(zh2)

    zo = (h1 * w7) + (h2 * w8) + bo

    y_pred = sigmoid(zo)

    # =================================================
    # OUTPUT SECTION
    # =================================================

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Prediction Probability",
            value=f"{y_pred:.4f}"
        )

    with col2:

        confidence = y_pred * 100

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    st.write("---")

    # =================================================
    # SURVIVAL RESULT
    # =================================================

    if y_pred > 0.5:

        st.success("✅ Passenger Survived")

        st.progress(float(y_pred))

    else:

        st.error("❌ Passenger Did Not Survive")

        st.progress(float(1 - y_pred))

    # =================================================
    # NETWORK DETAILS
    # =================================================

    with st.expander("View Neural Network Calculations"):

        st.write("### Hidden Layer Calculations")

        st.write(f"Hidden Neuron h1 Net Input: {zh1:.4f}")
        st.write(f"Hidden Neuron h2 Net Input: {zh2:.4f}")

        st.write(f"h1 Activation: {h1:.4f}")
        st.write(f"h2 Activation: {h2:.4f}")

        st.write("### Output Layer")

        st.write(f"Output Neuron Net Input: {zo:.4f}")
        st.write(f"Final Predicted Output: {y_pred:.4f}")

# =====================================================
# FOOTER
# =====================================================

st.write("---")

st.markdown("""
<center>

### Developed using Streamlit + Artificial Neural Network

</center>
""", unsafe_allow_html=True)
