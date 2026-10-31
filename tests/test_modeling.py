import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from market_regime.modeling import prepare_dataset, chronological_split, train_and_evaluate

def dataset(n=240):
    rng = np.random.default_rng(22)
    idx = pd.bdate_range("2021-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n,4)), index=idx, columns=list("ABCD"))
    y = pd.Series(np.where(np.arange(n)%3==0,"BULLISH",np.where(np.arange(n)%3==1,"SIDEWAYS","HIGH_RISK")), index=idx)
    return X,y

def test_chronological_split_preserves_order():
    X,y = dataset()
    a,b,c,d = chronological_split(X,y,.25)
    assert a.index.max() < b.index.min()
    assert len(b) == 60

def test_training_evaluates_all_models():
    X,y = dataset()
    results, models, *_ = train_and_evaluate(X,y,.25)
    assert len(results) == 3
    for result in results.values():
        assert 0 <= result["accuracy"] <= 1
        assert result["test_rows"] == 60

def test_small_dataset_rejected():
    X,y = dataset(50)
    with pytest.raises(ValueError):
        chronological_split(X,y)
