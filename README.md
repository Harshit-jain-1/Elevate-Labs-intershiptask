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

# Objective
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
# Objective
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

# Task 4: Logistic Regression: Breast Cancer Classification
# Objective
This project implements a binary classification model using Logistic Regression with L1 regularization to classify tumors as malignant or benign based on the Breast Cancer Wisconsin dataset.
# Key Steps Performed
. Build a binary classifier using Logistic Regression
. visualization the 'malignant', 'benign'
. Evaluate the model using common classification metrics
. Visualize performance using ROC Curve
. Explore effect of changing decision thresholds
. Understand the role of the sigmoid function

# Tash 5: Heart Disease Classification Using Decision Trees and Random Forests
# Objective
This project demonstrates how to build, evaluate, and interpret tree-based machine learning models for classification using the Heart Disease dataset.

# Key Steps Performed
. Train and evaluate Decision Tree and Random Forest classifiers on the Heart Disease dataset.
. Analyze overfitting by controlling tree depth.
. Interpret feature importance and evaluate with cross-validation.
. Evaluate both models using 5-fold cross-validation.

# Result
. Model Accuracy (5-Fold Cross-Validation)
. Overfitting Analysis
. Feature Importance (Random Forest)
. Evaluation Metrics

# Tsk 6: K-Nearest Neighbors (KNN) Classification
# Objective
This project demonstrates how to implement the K-Nearest Neighbors (KNN) algorithm for a classification task using the Iris dataset. It covers:

# Key Steps Performed
. Data preprocessing and normalization
. Model training using different values of K
. Model evaluation using accuracy, confusion matrix, and classification report
. Visualization of decision boundaries using PCA

# Result
. Model Accuracy (Different K Values)
. Evaluation Metrics (Best K = 3)
. Decision Boundary Visualization (PCA)

# Task 7: Support Vector Machines (SVM)

# Objective
This project demonstrates how to build a binary classification model using Support Vector Machines (SVMs) in Python. The workflow includes data preparation, model training (with linear and RBF kernels), dimensionality reduction for visualization, hyperparameter tuning, and model evaluation using cross-validation.

# Key Steps Performed
. Loaded and scaled data for binary classification
. Trained SVM models with linear and RBF kernel.
. Used PCA to reduce to 2D and plot decision boundaries
. Tuned C and gamma using grid search
. Checked model performance with cross-validation

# Result
. Best Hyperparameters (Grid Search)
. Cross-Validation Accuracy
. Decision Boundary Visualization (with PCA)

# Tash 8: Clustering with K-Means
# Obective
This project applies unsupervised learning using K-Means Clustering to segment customers based on their behavior. It uses the Mall Customer Segmentation Dataset to identify distinct customer groups for better marketing strategies.
# Key Steps Performed
. Elbow Plot: Visual tool to identify the optimal number of clusters
. Cluster Visualization: PCA-reduced 2D plot showing clusters with color-coding
. Silhouette Score: Metric to evaluate how well the data has been clustered

# Results
. Optimal number of clusters (K) is determined using the Elbow Method
. Customers are grouped based on age, income, and spending score
. Visual cluster boundaries and centroids help interpret customer segments
