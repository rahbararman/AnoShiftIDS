from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import config
from IDSAnoShift import data, utils


@pytest.fixture(scope="module")
def t_data():
    traffic_data = [
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.0",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.1",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.2",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.3",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.0",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.1",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.2",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.3",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.0",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.1",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.2",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "-1",
        ],
        [
            "c041",
            "other",
            "c263",
            "c363",
            "0",
            "0.3",
            "0.0",
            "0.41",
            "0",
            "0",
            "0.0",
            "0.0",
            "0.0",
            "SF",
            "1",
        ],
    ]
    col_names = utils.load_dict(Path(config.CONFIG_DIR, "col_names.json"))["names"]
    col_names = col_names + ["18"]
    traffic_data = pd.DataFrame(np.array(traffic_data), columns=col_names)
    traffic_data["18"] = pd.to_numeric(traffic_data["18"])
    return traffic_data


def test_get_data_splits_and_preprocess(t_data):
    t_data = t_data.copy()
    Y = t_data[["18"]]
    X = t_data.drop("18", axis=1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_data_splits(X, Y, train_size=0.7)
    assert len(X_train) == len(Y_train)
    assert len(X_val) == len(Y_val)
    assert len(X_test) == len(Y_test)
    assert len(X_train) / float(len(t_data)) == pytest.approx(0.7, abs=0.05)
    assert len(X_val) / float(len(t_data)) == pytest.approx(0.15, abs=0.05)
    assert len(X_test) / float(len(t_data)) == pytest.approx(0.15, abs=0.05)
    X_train_processed, Y_train_processed, numerical_cols, ohe_enc = data.get_preprocessed_train(
        X_train, Y_train
    )
    X_test_processed, Y_test_processed = data.get_test(X_test, ohe_enc, Y_test)
    assert len(X_train_processed.columns) == len(X_test_processed.columns)
