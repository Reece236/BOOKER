data {
  int<lower=1> N_players;
  int<lower=1> N_archetypes;
  array[N_players] int<lower=1, upper=N_archetypes> archetype;

  int<lower=1> N_teams;
  array[N_players] int<lower=1, upper=N_teams> team_id;
  vector[N_teams] team_wins;

  // Season totals
  vector[N_players] two_fieldGoals_made;
  vector[N_players] three_fieldGoals_made;
  vector[N_players] assists;
  vector[N_players] offensive_rebounds;
  vector[N_players] defensive_rebounds;
  vector[N_players] steals;
  vector[N_players] blocks;
  vector[N_players] turnovers;
  vector[N_players] personal_fouls;
  vector[N_players] field_attempts;
  vector[N_players] ft_attempts;
  vector[N_players] ft_made;
}

parameters {
  real alpha;
  matrix[12, N_archetypes] beta_by_archetype;

  real<lower=0, upper=50> points_per_win;
  real<lower=0> sigma_team;
}

transformed parameters {
  vector[N_players] player_point_value_total;
  vector[N_players] booker_war;

  for (i in 1:N_players) {
    int arch = archetype[i];
    real point_value = 
        alpha +
        beta_by_archetype[1, arch]  * two_fieldGoals_made[i] +
        beta_by_archetype[2, arch]  * three_fieldGoals_made[i] +
        beta_by_archetype[3, arch]  * assists[i] +
        beta_by_archetype[4, arch]  * offensive_rebounds[i] +
        beta_by_archetype[5, arch]  * defensive_rebounds[i] +
        beta_by_archetype[6, arch]  * steals[i] +
        beta_by_archetype[7, arch]  * blocks[i] +
        beta_by_archetype[8, arch]  * turnovers[i] +
        beta_by_archetype[9, arch]  * personal_fouls[i] +
        beta_by_archetype[10, arch] * field_attempts[i] +
        beta_by_archetype[11, arch] * ft_attempts[i] +
        beta_by_archetype[12, arch] * ft_made[i];

    player_point_value_total[i] = point_value;
  }
  
  booker_war = player_point_value_total / points_per_win;
}

model {
  alpha ~ normal(0, 0.5);

  for (a in 1:N_archetypes) {
    beta_by_archetype[1, a]  ~ normal(1.5, 0.2);    // two_fieldGoals_made
    beta_by_archetype[2, a]  ~ normal(2.5, 0.2);    // three_fieldGoals_made
    beta_by_archetype[3, a]  ~ normal(0.5, 0.2);    // assists
    beta_by_archetype[4, a]  ~ normal(0.5, 0.2);    // offensive_rebounds
    beta_by_archetype[5, a]  ~ normal(0.3, 0.2);    // defensive_rebounds
    beta_by_archetype[6, a]  ~ normal(0.75, 0.2);   // steals
    beta_by_archetype[7, a]  ~ normal(0.5, 0.2);    // blocks
    beta_by_archetype[8, a]  ~ normal(-1.5, 0.2);   // turnovers (negative)
    beta_by_archetype[9, a]  ~ normal(-0.2, 0.2);   // personal_fouls (negative)
    beta_by_archetype[10, a] ~ normal(-0.75, 0.2);  // field_attempts (negative)
    beta_by_archetype[11, a] ~ normal(-0.6, 0.5);   // ft_attempts (negative)
    beta_by_archetype[12, a] ~ normal(1.0, 0.5);    // ft_made
  }
  
  points_per_win ~ normal(15, 5); 
  sigma_team ~ normal(0, 2);

  for (t in 1:N_teams) {
    real team_war = 0;
    for (i in 1:N_players) {
      if (team_id[i] == t) {
        team_war += booker_war[i];
      }
    }
    team_wins[t] ~ normal(10 + team_war, sigma_team);
  }
}

generated quantities {
  vector[N_players] beta_two_fieldGoals_made;
  vector[N_players] beta_three_fieldGoals_made;
  vector[N_players] beta_assists;
  vector[N_players] beta_offensive_rebounds;
  vector[N_players] beta_defensive_rebounds;
  vector[N_players] beta_steals;
  vector[N_players] beta_blocks;
  vector[N_players] beta_turnovers;
  vector[N_players] beta_personal_fouls;
  vector[N_players] beta_field_attempts;
  vector[N_players] beta_ft_attempts;
  vector[N_players] beta_ft_made;
  vector[N_players] booker_war_posterior;

  for (i in 1:N_players) {
    int arch = archetype[i];
    beta_two_fieldGoals_made[i]   = beta_by_archetype[1, arch]  * two_fieldGoals_made[i];
    beta_three_fieldGoals_made[i]   = beta_by_archetype[2, arch]  * three_fieldGoals_made[i];
    beta_assists[i]              = beta_by_archetype[3, arch]  * assists[i];
    beta_offensive_rebounds[i]   = beta_by_archetype[4, arch]  * offensive_rebounds[i];
    beta_defensive_rebounds[i]   = beta_by_archetype[5, arch]  * defensive_rebounds[i];
    beta_steals[i]               = beta_by_archetype[6, arch]  * steals[i];
    beta_blocks[i]               = beta_by_archetype[7, arch]  * blocks[i];
    beta_turnovers[i]            = beta_by_archetype[8, arch]  * turnovers[i];
    beta_personal_fouls[i]       = beta_by_archetype[9, arch]  * personal_fouls[i];
    beta_field_attempts[i]       = beta_by_archetype[10, arch] * field_attempts[i];
    beta_ft_attempts[i]          = beta_by_archetype[11, arch] * ft_attempts[i];
    beta_ft_made[i]              = beta_by_archetype[12, arch] * ft_made[i];
    booker_war_posterior[i]      = booker_war[i];
  }
}
