library(jsonlite)
library(rstan)

stan_data <- jsonlite::fromJSON('full_model/stan_data.json')

model <- stan_model('full_model/booker.stan')

fit <- sampling(
  model,
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  refresh = 50
)

print(fit)

posterior <- rstan::extract(fit, pars = c('booker_war', 'beta_by_pos'))

print(posterior)