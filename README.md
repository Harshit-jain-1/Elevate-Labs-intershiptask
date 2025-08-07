# Titanic Data Cleaning & Preprocessing
  Tools: Python, Pandas, NumPy, Matplotlib/Seaborn, encoding, feature scaling 
# We cleaned and preprocessed the Titanic dataset to prepare it for machine learning.
# Key steps included:
  Handling missing values (Age, Embarked, etc.)
  Dropping or filling incomplete data
  Encoding categorical features (Sex, Embarked)
  Basic feature engineering 
  Data visualization for insights

The final cleaned dataset is ready for model training.

# Task 2: Exploratory Data Analysis (EDA)

#Objective
Perform EDA on the Titanic dataset to understand its structure, identify key patterns, and prepare for further modeling.
# Key Steps Performed
Summary Statistics
. Generated mean, median, standard deviation, etc.
. Checked for missing values and data types.

Visual Analysis
. Histograms for numeric distributions
. Boxplots to detect outliers
. Correlation matrix and pairplots to analyze relationships
. Count plots for categorical patterns

Patterns & Trends Identified
. Higher survival rate among females and 1st class passengers
. Younger passengers and those from Cherbourg had better survival chances
. Fare and Age contain outliers and missing values

Feature-Level Inferences
. Gender, class, and embarkation are strong survival indicators
. Cabin and Age require preprocessing due to missing data

# Task 3: House Price Prediction
#Objective
This project demonstrates how to implement Multiple Linear Regression using Python's scikit-learn to predict house prices based on multiple features from the Housing.csv dataset.
# Keys Steps Performed
Model Training
. Built a Multiple Linear Regression model with LinearRegression from sklearn

Model Evaluation
. MAE (Mean Absolute Error)
. MSE (Mean Squared Error)
. R² Score (Coefficient of Determination)

Visualization
. Actual vs Predicted Plot using seaborn
. Residual Plot to analyze prediction errors

Model Interpretation
. Printed model coefficients and intercept to understand how each feature affects house price
