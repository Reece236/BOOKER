library(jsonlite)
stan_data_final <- fromJSON('stan_data_final.json')

# --- Compile and Fit Stan Model ---
model <- stan_model("net_rating.stan")
fit <- sampling(
  model,
  data = stan_data_final,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  refresh = 100
)

player_data_filtered <- read.csv('player_season_stats_filtered.csv')



# --- 1. Extract all necessary posterior samples from the Stan fit object ---
posterior <- rstan::extract(fit, pars = c("war", "points_impact", "oreb_impact", "dreb_impact", 
                                          "blocks_impact", "assists_impact", "steals_impact", 
                                          "turnovers_impact", "fga_impact", "fta_impact"))



# --- 3. Create the results summary data frame ---
results <- player_data_filtered %>%
  select(firstName, lastName, season) %>%
  mutate(
    player_name = paste(firstName, lastName),
    
    war_mean           = apply(posterior$war, 2, mean)
  )

# --- 4. Identify and print the top players ---
top_20_players <- results %>%
  arrange(desc(war_mean)) %>%
  head(20) %>%
  mutate(rank = row_number())

print("=== TOP 20 PLAYER SEASONS BY WAR ===")
print(top_20_players %>% select(-firstName, -lastName))
