import numpy as np
from creditlens.data import FEATURES, make_dataset
from creditlens.train import run

def test_dataset():
    X, y, groups = make_dataset(n=4000, seed=1)
    assert X.shape == (4000, len(FEATURES))
    assert set(np.unique(y)) <= {0, 1}
    assert 0.05 < y.mean() < 0.5
    assert set(np.unique(groups)) == {"under_45", "45_plus"}

def test_model_has_signal_and_subgroup_report():
    report = run(seed=0)
    assert report["stronger_model"]["roc_auc"] >= 0.75, report
    assert report["subgroup_diagnostic"]["attribute"] == "age_band"
