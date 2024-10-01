import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
oppPer100 = pd.read_csv('data/opp-per100.csv')
playersPer100 = pd.read_csv('data/player-per100.csv')
teamPer100 = pd.read_csv('data/team-per100.csv')
teamSummaries = pd.read_csv('data/team-summaries.csv')
champsMvps = pd.read_csv('data/champs-mvps.csv')

# Drop unnecessary columns and filter for 1997-2024
def cleanAndFilterData():
    # Drop columns from opp-per100.csv
    oppPer100.drop(columns=['lg', 'mp'], inplace=True)
    
    # Drop columns from players-per100.csv
    playersPer100.drop(columns=['seas_id', 'player_id', 'birth_year', 'lg'], inplace=True)
    
    # Drop columns from team-per100.csv
    teamPer100.drop(columns=['lg', 'g', 'mp'], inplace=True)
    
    # Drop columns from team-summaries.csv
    teamSummaries.drop(columns=['lg', 'mov', 'pace', 'arena', 'attend', 'attend_g'], inplace=True)
    
    # Drop columns from champs-mvps.csv
    champsMvps.drop(columns=['index', 'Final Sweep ?'], inplace=True)

    # Filter data to include only seasons from 1997 to 2024
    oppPer100Filtered = oppPer100[(oppPer100['season'] >= 1997) & (oppPer100['season'] <= 2024)]
    playersPer100Filtered = playersPer100[(playersPer100['season'] >= 1997) & (playersPer100['season'] <= 2024)]
    teamPer100Filtered = teamPer100[(teamPer100['season'] >= 1997) & (teamPer100['season'] <= 2024)]
    teamSummariesFiltered = teamSummaries[(teamSummaries['season'] >= 1997) & (teamSummaries['season'] <= 2024)]
    champsMvpsFiltered = champsMvps[(champsMvps['Year'] >= 1997) & (champsMvps['Year'] <= 2024)]

    return oppPer100Filtered, playersPer100Filtered, teamPer100Filtered, teamSummariesFiltered, champsMvpsFiltered

# Execute the cleaning and filtering function
oppPer100Filtered, playersPer100Filtered, teamPer100Filtered, teamSummariesFiltered, champsMvpsFiltered = cleanAndFilterData()

# Merge oppPer100 and teamPer100 first on 'season' and 'team'
merged_data = pd.merge(oppPer100Filtered, teamPer100Filtered, on=['season', 'team', 'abbreviation', 'playoffs'], how='inner')
# Merge the result with teamSummaries
merged_data = pd.merge(merged_data, teamSummariesFiltered, on=['season', 'team', 'abbreviation', 'playoffs'], how='inner')
# Merge with champsMvps to include championship info
merged_data = pd.merge(merged_data, champsMvpsFiltered, left_on='season', right_on='Year', how='left')

# Create the target variable (binary: 1 if the team won the championship, 0 otherwise)
merged_data['is_champion'] = merged_data.apply(lambda row: 1 if row['NBA Champion'] == row['team'] else 0, axis=1)

# Columns like champion info, and MVP details should be removed as they can't be used for prediction
features_to_drop = [ 'NBA Champion', 'NBA Vice-Champion', 'MVP Name', 'MVP Team', 'Western Champion', 'Eastern Champion', 'Result', 'Year']
merged_data.drop(columns=features_to_drop, inplace=True)

# One-Hot Encode categorical variables (team, abbreviation)
merged_data = pd.get_dummies(merged_data, columns=['team', 'abbreviation'], drop_first=True)

train_data = merged_data[merged_data['season'] <= 2019]
test_data = merged_data[merged_data['season'] > 2019]

#Split into training and testing sets
X_train = train_data.drop(columns=['is_champion'])
y_train = train_data['is_champion']

X_test = test_data.drop(columns=['is_champion'])
y_test = test_data['is_champion']
