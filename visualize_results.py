import matplotlib.pyplot as plt

# Visualization function to show top n teams per year
def plot_top_teams_per_year(data, top_n=5):
    seasons = sorted(data['season'].unique())
    
    for season in seasons:
        season_data = data[data['season'] == season]
        # Get the top N teams for this season
        top_teams = season_data.nlargest(top_n, 'champion_prob')

        # Extract team names (from one-hot columns) and their probabilities
        team_names = top_teams[[col for col in top_teams.columns if col.startswith('team_')]].idxmax(axis=1).str.replace('team_', '')
        probabilities = top_teams['champion_prob']
        
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.barh(team_names, probabilities, color='skyblue')
        plt.xlabel('Championship Probability')
        plt.title(f'Top {top_n} Teams Predicted for {season}')
        plt.gca().invert_yaxis()  # Highest probabilities at the top
        plt.show()