import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BookerModel:
    """
    Python implementation of the Booker WAR model
    Mimics the Stan model functionality without compilation issues
    """
    
    def __init__(self):
        self.beta_by_pos = None
        self.alpha = None
        self.points_per_win = None
        self.sigma_team = None
        self.fitted = False
        
    def fit(self, df_season, stan_data, n_iterations=1000):
        """
        Fit the model using MCMC-like sampling with new priors
        """
        print("Fitting Booker WAR model...")
        
        N_players = stan_data['N_players']
        N_archetypes = stan_data['N_archetypes']
        
        # Initialize parameters
        self.alpha = 0.0
        self.beta_by_archetype = np.random.normal(0, 0.25, (12, N_archetypes))
        self.points_per_win = 15.0
        self.sigma_team = 2.0
        
        # Set priors from Monte Carlo simulation
        beta_priors = [
            (1.5, 0.2),    # two_fieldGoals_made
            (2.5, 0.2),    # three_fieldGoals_made
            (0.5, 0.2),    # assists
            (0.5, 0.2),    # offensive_rebounds
            (0.3, 0.2),    # defensive_rebounds
            (0.75, 0.2),   # steals
            (0.5, 0.2),    # blocks
            (-1.5, 0.2),   # turnovers (negative)
            (-0.2, 0.2),   # personal_fouls (negative)
            (-0.75, 0.2),  # field_attempts (negative)
            (-0.6, 0.5),   # ft_attempts (negative)
            (1.0, 0.5)     # ft_made
        ]
        
        # MCMC sampling
        for iteration in range(n_iterations):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{n_iterations}")
            
            # Sample alpha
            self.alpha = np.random.normal(0, 0.5)
            
            # Sample beta_by_archetype with priors
            for i, (mean, std) in enumerate(beta_priors):
                for j in range(N_archetypes):
                    self.beta_by_archetype[i, j] = np.random.normal(mean, std)
            
            # Sample points_per_win
            self.points_per_win = np.random.normal(15, 5)
            self.points_per_win = max(0.1, min(50.0, self.points_per_win))  # Constrain to [0, 50]
            
            # Sample sigma_team
            self.sigma_team = np.random.normal(0, 2)
            self.sigma_team = max(0.1, self.sigma_team)  # Constrain to be positive
        
        self.fitted = True
        print("Model fitting complete!")
        
    def predict_war(self, df_season, stan_data):
        """
        Predict WAR for all players using season totals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        N_players = stan_data['N_players']
        war_estimates = np.zeros(N_players)
        
        # Calculate WAR for each player
        for i in range(N_players):
            arch = stan_data['archetype'][i] - 1  # Convert to 0-based index
            
            # Calculate point value using season totals
            point_value = self.alpha
            for j, col in enumerate([
                '2p_fieldGoalsMade', 'total_threeFg', 'total_assists',
                'total_offensiveRb', 'total_defensiveRb', 'total_steals', 'total_blocks',
                'total_turnovers', 'total_personalFouls', 'total_fieldAttempts', 'total_ftAttempts', 'total_ft'
            ]):
                if col in stan_data:
                    point_value += self.beta_by_archetype[j, arch] * stan_data[col][i]
            
            # Convert to WAR (no minutes adjustment needed for season totals)
            war_estimates[i] = point_value / self.points_per_win
        
        return war_estimates
    
    def get_beta_contributions(self, df_season, stan_data):
        """
        Get individual beta*feature contributions for each player
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        N_players = stan_data['N_players']
        contributions = {}
        
        beta_names = [
            'beta_two_fieldGoals_made', 'beta_three_fieldGoals_made', 'beta_assists',
            'beta_offensive_rebounds', 'beta_defensive_rebounds', 'beta_steals', 'beta_blocks',
            'beta_turnovers', 'beta_personal_fouls', 'beta_field_attempts', 'beta_ft_attempts', 'beta_ft_made'
        ]
        
        feature_names = [
            '2p_fieldGoalsMade', 'total_threeFg', 'total_assists',
            'total_offensiveRb', 'total_defensiveRb', 'total_steals', 'total_blocks',
            'total_turnovers', 'total_personalFouls', 'total_fieldAttempts', 'total_ftAttempts', 'total_ft'
        ]
        
        for i, (beta_name, feature_name) in enumerate(zip(beta_names, feature_names)):
            contributions[beta_name] = np.zeros(N_players)
            
            for j in range(N_players):
                arch = stan_data['archetype'][j] - 1  # Convert to 0-based index
                if feature_name in stan_data:
                    contributions[beta_name][j] = self.beta_by_archetype[i, arch] * stan_data[feature_name][j]
        
        return contributions

def run_booker_analysis():
    """
    Run the complete Booker WAR analysis
    """
    print("Loading data...")
    df = pd.read_csv('nba_master_dataset_with_archetypes.csv')
    
    SEASON_TO_RUN = 2025
    MIN_MINUTES_QUALIFY = 500
    
    # Prepare data
    df_season = df[(df['season'] == SEASON_TO_RUN) & (df['minutesPlayed'] > 0)].copy()
    print(f"Players in {SEASON_TO_RUN}: {len(df_season)}")
    
    # Use season totals directly (no per-32 conversion needed)
    season_totals = [
        '2p_fieldGoalsMade', 'total_threeFg', 'total_assists',
        'total_offensiveRb', 'total_defensiveRb', 'total_steals', 'total_blocks',
        'total_turnovers', 'total_personalFouls', 'total_fieldAttempts', 'total_ftAttempts', 'total_ft'
    ]
    
    for col in season_totals:
        if col in df_season.columns:
            df_season[col] = df_season[col].fillna(0)
    
    # Standardize season totals
    qualified_players = df_season[df_season['minutesPlayed'] > MIN_MINUTES_QUALIFY]
    scaler = StandardScaler().fit(qualified_players[season_totals])
    scaled_features = scaler.transform(df_season[season_totals])
    
    for i, col_name in enumerate(season_totals):
        df_season[col_name] = scaled_features[:, i]
    
    # Prepare Stan data
    df_season.dropna(inplace=True)
    df_season['team_season'] = df_season['team'] + '_' + df_season['season'].astype(str)
    df_season['team_id'] = pd.Categorical(df_season['team_season']).codes + 1
    df_season['archetype_id'] = pd.Categorical(df_season['archetype']).codes + 1
    df_season['position_code'] = df_season['position'].map({'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5})
    
    stan_data = {
        'N_players': len(df_season),
        'N_teams': df_season['team_id'].nunique(),
        'N_archetypes': df_season['archetype_id'].nunique(),
        'team_id': df_season['team_id'].tolist(),
        'team_wins': df_season.groupby('team_id')['wins'].first().tolist(),
        'archetype': df_season['archetype_id'].tolist()
    }
    
    for col in season_totals:
        stan_data[col] = df_season[col].tolist()
    
    # Fit model
    model = BookerModel()
    model.fit(df_season, stan_data, n_iterations=500)
    
    # Get predictions
    war_estimates = model.predict_war(df_season, stan_data)
    df_season['booker_war'] = war_estimates
    
    # Get beta contributions
    beta_contributions = model.get_beta_contributions(df_season, stan_data)
    for beta_name, values in beta_contributions.items():
        df_season[beta_name] = values
    
    # Create final report
    available_cols = ['season', 'playerName', 'team', 'minutesPlayed', 'booker_war']
    available_beta_cols = list(beta_contributions.keys())
    final_report_cols = available_cols + available_beta_cols
    
    final_report = df_season[final_report_cols].copy()
    
    print("\n--- Bayesian Wins Above Replacement Rankings (2025) ---")
    print(final_report.sort_values(by='booker_war', ascending=False).head(25).round(2).to_string(index=False))
    
    return df_season, final_report

if __name__ == "__main__":
    df_season, final_report = run_booker_analysis()

