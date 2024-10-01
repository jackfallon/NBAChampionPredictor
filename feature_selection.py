import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to find and plot the most important features
def plot_feature_importance(model, feature_names, top_n=10):
    
    # Get feature importances from the RandomForest model
    importances = model.feature_importances_
    
    # Create a DataFrame with feature names and their importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Plot the top N important features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'], color='lightblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Important Features for Predicting NBA Championship')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()
    
    return top_features


