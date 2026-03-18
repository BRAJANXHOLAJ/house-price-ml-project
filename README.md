House Price Prediction  Machine Learning Project
1. Project Overview

This project builds a machine learning system that predicts house prices based on structural and location characteristics of residential properties.

The objective is to demonstrate a complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, and deployment through an interactive web interface.

2. Dataset

The project uses the Ames Housing Dataset, a well-known dataset frequently used in machine learning and data science research.

Dataset characteristics:

Attribute	Value
Houses	2930
Features	264
Target Variable	SalePrice

The dataset contains detailed information about each property, including:

lot size

construction year

living area

garage capacity

basement features

neighborhood information

structural characteristics

3. Machine Learning Models

Several regression algorithms were trained and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

The goal was to compare different modeling approaches and identify the model with the best predictive performance.

4. Model Performance

After evaluation using RMSE and R² score, the Random Forest model produced the best results.

Model	R²	RMSE
Linear Regression	0.894	~29,100
Decision Tree	0.844	~35,400
Random Forest	0.912	~26,500

Random Forest captured nonlinear relationships between house features and price more effectively than the other models.

5. Key Predictive Features

Feature importance analysis identified the most influential variables affecting house prices.

Top predictors include:

Overall Quality of the house

Above Ground Living Area

Garage Capacity

Total Basement Area

Year Built

These features strongly influence property value and align with real-world housing market expectations.

6. Interactive Web Application

A simple web interface was developed using Streamlit to allow users to interact with the trained model.

Users can input house characteristics such as:

quality rating

living area

garage size

construction year

basement area

The application then estimates the predicted house price using the trained Random Forest model.