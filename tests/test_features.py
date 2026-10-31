import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
from market_regime.features import build_features, make_regime_labels

def sample_prices(n=400):
    rng = np.random.default_rng(11)
    returns = rng.normal(.0003, .01, (n,3))
    return pd.DataFrame((1+returns).cumprod(axis=0)*[100,120,80],
                        index=pd.bdate_range("2020-01-01", periods=n), columns=["A","B","C"])

def test_features_shape_and_no_infinite_values():
    p = sample_prices()
    f = build_features(p)
    assert len(f) == len(p)
    assert np.isfinite(f.select_dtypes("number").dropna().to_numpy()).all()

def test_labels_are_documented_classes():
    labels = make_regime_labels(sample_prices())
    assert set(labels.dropna().unique()).issubset({"HIGH_RISK","BULLISH","BEARISH","SIDEWAYS"})
    assert labels.notna().sum() > 100
