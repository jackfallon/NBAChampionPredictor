from data_prep import *
import visualize_results
from feature_selection import plot_feature_importance

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Implement Bagging with Random Forest as the base estimator
bagging_model = BaggingClassifier(estimator=rf_model, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_model.predict(X_test)

# Predict probabilities for championship
y_pred_probs = bagging_model.predict_proba(X_test)[:, 1]
test_data.loc[:, 'champion_prob'] = y_pred_probs

# Visualize the results
visualize_results.plot_top_teams_per_year(test_data, top_n=5)

# Extract feature names from the training data
feature_names = X_train.columns
# Use the Random Forest model to find the most important features
top_features = plot_feature_importance(rf_model, feature_names, top_n=5)
