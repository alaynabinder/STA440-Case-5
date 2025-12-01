data {
  int<lower = 1> N;
  int<lower = 1> n;
  int<lower = 1> p;
  int Y[N];               
  matrix[N, p] X;
  int<lower = 1, upper = n> Ids[N];
}
transformed data {
  matrix[N, p] X_centered;
  row_vector[p] X_bar;
  for (i in 1:p) {
    X_bar[i] = mean(X[, i]);
    X_centered[, i] = X[, i] - X_bar[i];
  }
}
parameters {
  real alpha;
  vector[p] beta;
  vector[n] theta;
  real<lower = 0> tau;
}
model {
  vector[N] mu = rep_vector(0.0, N);
  mu += alpha;
  for (i in 1:N) {
    mu[i] += X_centered[i, ] * beta + theta[Ids[i]];
  }
  target += bernoulli_logit_lpmf(Y | mu); 
  target += normal_lpdf(theta | 0, tau);
  target += normal_lpdf(alpha | 0, 10);
  target += normal_lpdf(beta | 0, 10);
  target += normal_lpdf(tau | 0, 3);
}
generated quantities {
  vector[N] Y_pred;
  vector[N] log_lik;
  vector[N] mu = rep_vector(0.0, N);
  mu += alpha;
  for (i in 1:N) {
    mu[i] += X_centered[i, ] * beta + theta[Ids[i]];
    Y_pred[i] = bernoulli_logit_rng(mu[i]);
    log_lik[i] = bernoulli_logit_lpmf(Y[i] | mu[i]);
  }
}
