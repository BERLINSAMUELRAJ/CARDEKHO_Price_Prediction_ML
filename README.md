# CarDekho - Used Car Price Prediction

## Project Overview
The objective of this project is to develop an accurate, user-friendly machine learning system for predicting the prices of used cars based on various features. The project leverages historical data collected from CarDekho, including details like make, model, year of manufacture, fuel type, transmission, location, and car specifications.  

The project includes **data preprocessing, exploratory data analysis, model development, evaluation, optimization, and deployment** using a Streamlit application, providing a seamless experience for users to estimate car prices in real time.  

---

## Skills & Takeaways
- Data Cleaning and Preprocessing  
- Exploratory Data Analysis (EDA)  
- Machine Learning Model Development  
- Price Prediction Techniques  
- Model Evaluation and Optimization  
- Model Deployment  
- Streamlit Application Development  
- Documentation and Reporting  

---

## Domain
- Automotive Industry  
- Data Science  
- Machine Learning  

---

## Problem Statement
Imagine working as a data scientist at CarDekho, tasked with improving the customer experience and streamlining the pricing process. The goal is to create a **machine learning model integrated into a Streamlit app** that predicts used car prices accurately and efficiently.  

---

## Project Scope
- Collect historical used car data from different cities.  
- Develop a machine learning model to predict car prices using key features.  
- Deploy a **Streamlit application** for interactive, real-time price estimation.  

---

## Approach

### 1️⃣ Data Processing
- **Import & Concatenate:** Combine multiple city datasets into a structured dataset with a `City` column.  
- **Handle Missing Values:** Impute missing numerical values using median, categorical values using mode.  
- **Standardize Formats:** Remove units from numeric columns and convert to appropriate data types.  
- **Encode Categorical Variables:** One-hot encoding for nominal features, label encoding for ordinal.  
- **Normalize Numerical Features:** Standard scaling for features where necessary.  
- **Remove Outliers:** Identify and cap outliers using IQR or Z-score methods.  

### 2️⃣ Exploratory Data Analysis (EDA)
- **Descriptive Statistics:** Mean, median, mode, standard deviation.  
- **Data Visualization:** Scatter plots, histograms, box plots, and correlation heatmaps to identify patterns.  
- **Feature Selection:** Identify important features using correlation analysis, model feature importance, and domain knowledge.  

### 3️⃣ Model Development
- **Train-Test Split:** 80-20 split for training and evaluation.  
- **Model Selection:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting.  
- **Model Training:** Trained models on the preprocessed dataset.  
- **Hyperparameter Tuning:** RandomizedSearchCV for optimal Random Forest parameters.  

### 4️⃣ Model Evaluation
- **Metrics Used:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.  
- **Model Comparison:** Random Forest performed the best, providing the lowest RMSE and highest R².  

### 5️⃣ Deployment
- **Streamlit Application:** User-friendly interface to input car details and get real-time price predictions.  
- **Instructions & Error Handling:** The app guides the user and fills default values for non-critical features.  

---

## Observations
- **Linear Regression** performs poorly — likely due to non-linear relationships between features and price.  
- **Decision Tree** improves performance significantly, capturing non-linear patterns.  
- **Random Forest** is the best model, with the highest R² and lowest RMSE.  
- **Gradient Boosting** performs close to Random Forest but slightly worse in this run.  

---

## Results
- Functional **Random Forest model** for predicting used car prices.  
- Streamlit app deployed for real-time prediction.  
- Comprehensive **EDA and feature analysis** report.  
- Detailed documentation explaining methodology, models, and results.  

---

## Project Deliverables
- Preprocessing and model development source code.  
- Documentation detailing methodology, models, and evaluation results.  
- Visualizations and EDA analysis reports.  
- Deployed Streamlit app for price prediction.  
- Justification for approach and model selection.  

---

## Evaluation Metrics
- **Model Performance:** MAE, MSE, R²  
- **Data Quality:** Completeness and accuracy of preprocessed data  
- **Application Usability:** User satisfaction and feedback on the Streamlit app  
- **Documentation:** Clarity and completeness of project report and code comments  

---

## Technical Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Streamlit  
- **Techniques:** Data Cleaning, EDA, Regression Models, Hyperparameter Tuning  
- **Deployment:** Streamlit  

---

## Dataset
- **Source:** CarDekho  
- **Description:** Multiple Excel files, each representing a city, containing car features, specifications, and pricing information.  
- **Preprocessing:** Structured, missing value handling, encoding categorical variables, and normalizing numerical features.  

---

## Screenshots (Optional)
You can include screenshots of your Streamlit application or example predictions here to make the repository visually appealing.

---

This README provides a complete overview of your **CarDekho Used Car Price Prediction** project, making it professional and easy to understand for trainers, peers, or employers.
