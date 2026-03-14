import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("models/random_forest_model.pkl")
features = joblib.load("models/feature_names.pkl")

st.title("House Price Prediction")

st.write("Enter house characteristics to estimate the price.")

# User inputs
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (square feet)", 300, 6000, 1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
year_built = st.number_input("Year Built", 1800, 2025, 2000)
total_bsmt = st.number_input("Total Basement Area", 0, 3000, 800)

# Create feature vector
input_data = pd.DataFrame(0, index=[0], columns=features)

input_data["Overall Qual"] = overall_qual
input_data["Gr Liv Area"] = gr_liv_area
input_data["Garage Cars"] = garage_cars
input_data["Year Built"] = year_built
input_data["Total Bsmt SF"] = total_bsmt

# Prediction button
if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: ${int(prediction[0]):,}")