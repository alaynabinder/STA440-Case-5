data {
  int<lower = 1> N;                // number of observations
  int<lower = 1> n;                // number of groups for theta
  int<lower = 1> p;                // number of predictors

  // Missing-data bookkeeping
  int<lower = 0> N_miss;           // number of missing X entries
  int<lower = 1, upper = N> miss_row[N_miss];
  int<lower = 1, upper = p> miss_col[N_miss];

  matrix[N, p] X_obs;              // matrix with observed values; missing positions unused
  int<lower=0,upper=1> is_missing[N, p];

  int<lower=0,upper=1> Y[N];       // binary outcome
  int<lower = 1, upper = n> Ids[N];
}

parameters {
  real alpha;
  vector[p] beta;
  vector[n] theta;
  vector[n] age_impute;
  real<lower = 0> tau;

  // Missing covariates
  vector[N_miss] X_miss;           // imputed missing values
}

transformed parameters {
  matrix[N, p] X_full = X_obs;     // start from observed values

  // Insert missing values
  for (k in 1:N_miss) {
    X_full[ miss_row[k], miss_col[k] ] = X_miss[k];
  }

  // Center X after imputation
  matrix[N, p] X_centered;
  row_vector[p] X_bar;

  for (j in 1:p) {
    X_bar[j] = mean(X_full[, j]);
    X_centered[, j] = X_full[, j] - X_bar[j];
  }
}

model {
  vector[N] mu = rep_vector(0.0, N);

  // Priors
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  theta ~ normal(0, tau);
  tau   ~ normal(0, 3);
  age_impute ~ normal(68,5);
  // Prior for missing X values (adjust if needed)
  

  // Linear predictor
  mu += alpha;
  for (i in 1:N) {
    X_miss[i] ~ normal(age_impute[Ids[i]], 5);
    mu[i] += X_centered[i, ] * beta + theta[Ids[i]];
  }

  // Likelihood
  Y ~ bernoulli_logit(mu);
}

generated quantities {
  vector[N] Y_pred;
  vector[N] log_lik;

  vector[N] mu = rep_vector(0.0, N);
  for (i in 1:N) {
    mu[i] += alpha + X_centered[i, ] * beta + theta[Ids[i]];
    Y_pred[i] = bernoulli_logit_rng(mu[i]);
    log_lik[i] = bernoulli_logit_lpmf(Y[i] | mu[i]);
  }
}
