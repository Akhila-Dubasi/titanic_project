import streamlit as st
import math
import json

# -----------------------------
# LOAD SAVED WEIGHTS
# -----------------------------

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

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Titanic ANN Prediction")

x1 = st.slider("Pclass", 0.0, 1.0, 0.2)
x2 = st.slider("Age", 0.0, 1.0, 0.24)
x3 = st.slider("Fare", 0.0, 1.0, 0.80)

# -----------------------------
# SIGMOID
# -----------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# -----------------------------
# FORWARD PROPAGATION
# -----------------------------

zh1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + bh1
zh2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + bh2

h1 = sigmoid(zh1)
h2 = sigmoid(zh2)

zo = (h1 * w7) + (h2 * w8) + bo

y_pred = sigmoid(zo)

# -----------------------------
# OUTPUT
# -----------------------------

st.subheader("Prediction")

st.write("Predicted Output:", round(y_pred, 4))

if y_pred > 0.5:
    st.success("Passenger Survived")
else:
    st.error("Passenger Did Not Survive")
