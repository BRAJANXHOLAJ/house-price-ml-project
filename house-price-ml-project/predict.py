import joblib
import pandas as pd


# Load trained model and features


model = joblib.load("models/random_forest_model.pkl")
features = joblib.load("models/feature_names.pkl")

print("House Price Prediction System\n")

# Ask user inputs

overall_qual = int(input("Overall Quality (1-10): "))
gr_liv_area = int(input("Living Area (square feet): "))
garage_cars = int(input("Garage Capacity (cars): "))
year_built = int(input("Year Built: "))
total_bsmt = int(input("Total Basement Area: "))


# Create full feature vector

# create dataframe with all features initialized to 0
input_data = pd.DataFrame(0, index=[0], columns=features)

# fill only the known inputs
input_data["Overall Qual"] = overall_qual
input_data["Gr Liv Area"] = gr_liv_area
input_data["Garage Cars"] = garage_cars
input_data["Year Built"] = year_built
input_data["Total Bsmt SF"] = total_bsmt



# Predict price


prediction = model.predict(input_data)

print("\nEstimated House Price: $", int(prediction[0]))