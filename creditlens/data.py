"""Deterministic, domain-specific synthetic dataset for CreditLens-Risk — Lending Research Prototype."""

from __future__ import annotations
import numpy as np

FEATURES = ['income', 'loan', 'dti', 'history', 'delinquencies', 'employment', 'util', 'age']
GROUP_NAME = 'age_band'
GROUP_SPLIT = False

def make_dataset(n=2400, seed=42):
    if n < 80:
        raise ValueError("n must be at least 80 for a useful holdout")
    rng = np.random.default_rng(seed)
    income = np.clip(rng.normal(60, 28, n), 12, 250)
    loan = np.clip(rng.normal(18, 12, n), 1, 80)
    dti = np.clip(rng.normal(.32, .14, n), .02, .95)
    history = np.clip(rng.normal(9, 5, n), 0, 35)
    delinquencies = rng.poisson(.5, n).astype(float)
    employment = np.clip(rng.normal(6, 4, n), 0, 35)
    util = np.clip(rng.normal(.45, .22, n), 0, 1)
    age = rng.integers(21, 75, n).astype(float)
    z = -2.6 + 4.3*dti + .95*delinquencies + 2.7*util + .028*loan - .020*income - .070*history - .060*employment - .012*np.maximum(age-30, 0)
    p = 1/(1+np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    groups = np.where(age < 45, "under_45", "45_plus")
    X = np.column_stack([income, loan, dti, history, delinquencies, employment, util, age]).astype(float)
    if X.shape != (n, len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("generator produced invalid feature matrix")
    if np.asarray(groups).shape != (n,):
        raise ValueError("generator produced invalid group vector")
    return X, y.astype(int), np.asarray(groups)
