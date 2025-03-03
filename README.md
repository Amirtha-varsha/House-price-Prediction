# House Price Prediction

## Overview
This project is a House Price Prediction system that estimates the price of a house based on various factors such as area, number of BHKs, number of bathrooms, and location. The model utilizes Machine Learning (ML) techniques to predict house prices based on historical data.

## Tech Stack Used
- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn, Pandas, NumPy  
- **Dataset:** Bangalore House Price Dataset  

## Features
- Users can input house details such as area (in sqft), BHK, bathrooms, and location.
- Predicts the house price in INR Lakhs.
- Displays results dynamically in the frontend.

## Project Workflow

### 1. **Data Collection & Preprocessing**
- Collected a dataset containing house prices in Bangalore.
- Cleaned missing values, removed outliers, and converted categorical data into numerical form.
- Applied One-Hot Encoding for categorical variables (Location).
- Normalized numerical features for better model performance.

### 2. **Model Training & Evaluation**
- Selected **Linear Regression** as it provided a good balance between performance and interpretability.
- Achieved an **R² score of 0.86**, indicating a strong correlation between input features and house prices.

### 3. **Backend Development**
- Developed a Flask API to handle user inputs and return predictions.
- Integrated the trained model into the Flask backend.


## Challenges Faced & Solutions

### **1. Handling Categorical Data in the Model**
- **Issue:** The ML model couldn't directly process location data.  
- **Solution:** Used **One-Hot Encoding** to convert categorical locations into numerical features.

### **2. Model Accuracy Improvements**
- **Issue:** Initial predictions were inaccurate due to outliers and feature imbalance.  
- **Solution:** Applied **feature scaling** and **outlier removal techniques** to improve accuracy.

### **3. Frontend-Backend Integration Issues**
- **Issue:** User inputs were not properly passed between the Flask backend and the frontend.  
- **Solution:** Debugged Flask routes, ensured correct form submission, and used AJAX for seamless data transfer.


## Output Screenshot
# ** Index **
artifacts/index.png
# ** Home **
artifacts/home.png
# ** Predicted Price **
artifacts/predicted price.png
