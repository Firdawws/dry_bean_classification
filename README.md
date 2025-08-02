### Dry Bean Classification System using ML 
### Overview
This project presents a machine learning-based solution for the automated classification of dry bean types based on shape-related morphological features. 
Traditional manual sorting is labor-intensive and error-prone — this system offers an efficient, accurate, and scalable alternative using supervised learning.

----
## Dataset Details
Dataset: Dry Bean Dataset (UCI Machine Learning Repository)
Samples: 13,611
Classes: 7 Dry Bean species
(e.g., Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, and Sira)

----

## Project Highlights
✔️ Exploratory Data Analysis (EDA)
Conducted in-depth data analysis using distribution plots, boxplots, pairplots, and correlation heatmaps to detect patterns, anomalies, and feature relationships.

✔️ Model Training & Performance Comparison
Trained and benchmarked five classifiers:

Logistic Regression (top performer)
Decision Tree
Random Forest
Gradient Boosting
XGBoost

✔️ Feature Selection with RFE
Employed Recursive Feature Elimination to pinpoint the most impactful features for classification.

✔️ Dimensionality Reduction via PCA
Used Principal Component Analysis to compress feature space and visualize separability between bean classes in 2D.

✔️ Cross-Validation & Evaluation
Applied stratified cross-validation and visualized model results using confusion matrices and per-class accuracy scores.

✔️ Hyperparameter Optimization
Optimized Gradient Boosting and XGBoost models using GridSearchCV and RandomizedSearchCV for enhanced accuracy.

✔️ Model Evaluation
Confusion Matrices
Classification Reports
Per-class accuracy scores
Visual assessment of misclassifications and overall balance.

----
### 🏆 Results Summary
Top Features: Solidity, ShapeFactor1, ShapeFactor4, Eccentricity
Best Model: Logistic Regression
Highest accuracy
Fast inference time
Transparent and interpretable
Well-suited for deployment in real-time systems 

---

## Deployment
The final model was deployed using Streamlit
To launch the Streamlit web interface, use the following command:

```bash
streamlit run app.py

```


## Tools & Technologies used:
Python Libraries: Pandas, NumPy, Scikit-learn

ML Models: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, Decision Tree

Visualization: Matplotlib, Seaborn

Development Environment: Jupyter Notebook

Deployment: Streamlit for building the user interface




