data {
  int<lower=1> N_players;
  int<lower=1> N_archetypes;
  array[N_players] int<lower=1> position;
  vector[N_players] total_minutes_played;

  int<lower=1> N_teams;
  array[N_players] int<lower=1, upper=N_teams> team_id;
  vector[N_teams] team_wins;

  vector[N_players] z_points;
  vector[N_players] z_assists;
  vector[N_players] z_orebounds;
  vector[N_players] z_turnovers;
  vector[N_players] z_missed_fga;
  vector[N_players] z_missed_fta;
  vector[N_players] z_drebounds;
  vector[N_players] z_steals;
  vector[N_players] z_blocks;
  vector[N_players] z_fouls;
}

parameters {
  real alpha;
  matrix[10, N_archetypes] beta_by_pos;

  real<lower=0, upper=5> points_per_win;
  real<lower=0> sigma_team;
}

transformed parameters {
  vector[N_players] player_point_value_total;
  vector[N_players] war;

  for (i in 1:N_players) {
    int pos = position[i];
    real point_value_z = 
        alpha +
        beta_by_pos[1, pos] * z_points[i] +
        beta_by_pos[2, pos] * z_assists[i] +
        beta_by_pos[3, pos] * z_orebounds[i] +
        beta_by_pos[4, pos] * z_turnovers[i] +
        beta_by_pos[5, pos] * z_missed_fga[i] +
        beta_by_pos[6, pos] * z_missed_fta[i] +
        beta_by_pos[7, pos] * z_drebounds[i] +
        beta_by_pos[8, pos] * z_steals[i] +
        beta_by_pos[9, pos] * z_blocks[i] +
        beta_by_pos[10, pos] * z_fouls[i];

    player_point_value_total[i] = point_value_z * (total_minutes_played[i] / 2000.0);
  }
  
  war = player_point_value_total / points_per_win;
}

model {
  alpha ~ normal(0, 0.5);

  for (p in 1:N_archetypes) {
    beta_by_pos[1, p] ~ normal(3, 0.25);   // points
    beta_by_pos[2, p] ~ normal(0.75, 0.25);   // assists
    beta_by_pos[3, p] ~ normal(0.7, 0.25);   // orebounds
    beta_by_pos[4, p] ~ normal(-0.5, 0.25);  // turnovers
    beta_by_pos[5, p] ~ normal(-0.4, 0.25);  // missed_fga
    beta_by_pos[6, p] ~ normal(-0.3, 0.25);  // missed_fta
    beta_by_pos[7, p] ~ normal(0.3, 0.25);   // drebounds
    beta_by_pos[8, p] ~ normal(0.2, 0.25);   // steals
    beta_by_pos[9, p] ~ normal(0.3, 0.25);   // blocks
    beta_by_pos[10, p] ~ normal(-0.2, 0.25); // fouls
  }
  
  points_per_win ~ normal(.5, .5); 
  sigma_team ~ normal(0, 2);

  for (t in 1:N_teams) {
    real team_war = 0;
    for (i in 1:N_players) {
      if (team_id[i] == t) {
        team_war += war[i];
      }
    }
    team_wins[t] ~ normal(10 + team_war, sigma_team);
  }
}

generated quantities {
  vector[N_players] beta_points_z;
  vector[N_players] beta_assists_z;
  vector[N_players] beta_orebounds_z;
  vector[N_players] beta_missed_fga_z;
  vector[N_players] beta_missed_fta_z;
  vector[N_players] beta_drebounds_z;
  vector[N_players] beta_steals_z;
  vector[N_players] beta_blocks_z;
  vector[N_players] beta_fouls_z;
  vector[N_players] booker_war;

  for (i in 1:N_players) {
    int pos = position[i];
    beta_points_z[i]      = beta_by_pos[1, pos]  * z_points[i];
    beta_assists_z[i]     = beta_by_pos[2, pos]  * z_assists[i];
    beta_orebounds_z[i]   = beta_by_pos[3, pos]  * z_orebounds[i];
    beta_missed_fga_z[i]  = beta_by_pos[5, pos]  * z_missed_fga[i];
    beta_missed_fta_z[i]  = beta_by_pos[6, pos]  * z_missed_fta[i];
    beta_drebounds_z[i]   = beta_by_pos[7, pos]  * z_drebounds[i];
    beta_steals_z[i]      = beta_by_pos[8, pos]  * z_steals[i];
    beta_blocks_z[i]      = beta_by_pos[9, pos]  * z_blocks[i];
    beta_fouls_z[i]       = beta_by_pos[10, pos] * z_fouls[i];
  }
}
