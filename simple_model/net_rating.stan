data {
  int<lower=1> N_players;

  // The Target Variable
  vector[N_players] plus_minus_p36;
  vector[N_players] total_minutes;

  // Predictor Variables
  vector[N_players] points_p36;
  vector[N_players] assists_p36;
  vector[N_players] oreb_p36;
  vector[N_players] dreb_p36;
  vector[N_players] steals_p36;
  vector[N_players] blocks_p36;
  vector[N_players] turnovers_p36;
  vector[N_players] fga_p36;
  vector[N_players] fta_p36;
}

parameters {
  real alpha;
  real beta_pts;
  real beta_ast;
  real beta_oreb;
  real beta_dreb;
  real beta_stl;
  real beta_blk;
  real beta_tov;
  real beta_fga;
  real beta_fta;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(0, 2.0); 
  
  beta_pts ~ normal(0.7, 0.2);
  beta_ast ~ normal(0.3, 0.2);
  beta_oreb ~ normal(0.7, 0.3);
  beta_dreb ~ normal(0.3, 0.3);
  beta_stl ~ normal(1.1, 0.3);
  beta_blk ~ normal(0.6, 0.3);
  beta_tov ~ normal(-1.1, 0.2);
  beta_fga ~ normal(-0.9, 0.2);
  beta_fta ~ normal(-0.4, 0.2);
  
  sigma ~ exponential(1);

  vector[N_players] mu;
  mu = alpha +
       beta_pts * points_p36 +
       beta_ast * assists_p36 +
       beta_oreb * oreb_p36 +
       beta_dreb * dreb_p36 +
       beta_stl * steals_p36 +
       beta_blk * blocks_p36 +
       beta_tov * turnovers_p36 +
       beta_fga * fga_p36 +
       beta_fta * fta_p36;
       
  plus_minus_p36 ~ normal(mu, sigma);
}

generated quantities {

  vector[N_players] player_bpm_p36;
  vector[N_players] war;
  
  vector[N_players] points_impact;
  vector[N_players] oreb_impact;
  vector[N_players] dreb_impact;
  vector[N_players] blocks_impact;
  vector[N_players] assists_impact;
  vector[N_players] steals_impact;
  vector[N_players] turnovers_impact;
  vector[N_players] fga_impact;
  vector[N_players] fta_impact;

  player_bpm_p36 = alpha +
                   beta_pts * points_p36 +
                   beta_ast * assists_p36 +
                   beta_oreb * oreb_p36 +
                   beta_dreb * dreb_p36 +
                   beta_stl * steals_p36 +
                   beta_blk * blocks_p36 +
                   beta_tov * turnovers_p36 +
                   beta_fga * fga_p36 +
                   beta_fta * fta_p36;

  points_impact    = beta_pts * points_p36;
  oreb_impact      = beta_oreb * oreb_p36;
  dreb_impact      = beta_dreb * dreb_p36;
  blocks_impact    = beta_blk * blocks_p36;
  assists_impact   = beta_ast * assists_p36;
  steals_impact    = beta_stl * steals_p36;
  turnovers_impact = beta_tov * turnovers_p36;
  fga_impact       = beta_fga * fga_p36;
  fta_impact       = beta_fta * fta_p36;

  real replacement_level_p36 = -1.5;
  real points_per_win = 50.0;
  
  war = ((player_bpm_p36 - replacement_level_p36) / 36) .* total_minutes / points_per_win;
}
