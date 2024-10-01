from data_prep import *
import visualize_results
from feature_selection import plot_feature_importance

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict the probabilities for the test set
y_pred_probs = rf_model.predict_proba(X_test)[:, 1]  # Probability of being the champion (class 1)

test_data.loc[:, 'champion_prob'] = y_pred_probs

# Ensure grouping columns are not included in the operation
predictions_by_season = test_data.groupby('season', group_keys=False).apply(lambda x: x.loc[x['champion_prob'].idxmax()])

# Print predicted champions for each season
predicted_champions = predictions_by_season[[col for col in predictions_by_season.columns if col.startswith('team_') or col == 'season' or col == 'champion_prob']]

# Extract the actual team name by reversing the one-hot encoding (team_* columns)
predicted_champions['predicted_team'] = predicted_champions[[col for col in predicted_champions.columns if col.startswith('team_')]].idxmax(axis=1).str.replace('team_', '')

# Display the predicted champions for 2021, 2022, 2023, etc.
print("Predicted NBA Champions (2021, 2022, 2023, etc.):")
print(predicted_champions[['season', 'champion_prob', 'predicted_team']])

# Visualize the results
visualize_results.plot_top_teams_per_year(test_data, top_n=5)

# Extract feature names from the training data
feature_names = X_train.columns

# Call the function to plot and display top 10 important features
top_features = plot_feature_importance(rf_model, feature_names, top_n=10)

# Print the top important features
print(top_features)
