from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from config import config
from IDSAnoShift import data, utils


def train(args, df=None, label_col="18", train_size=0.8, pos_label=-1):
    """This function trains a model given a dataset.

    Args:
        args (Namespace): training parameters
        df (pd.DataFrame, optional): DataFrame containing a dataset. Defaults to None.
        label_col (str, optional): Name of the label column in df. Defaults to '18'.
        train_size (float, optional): size of the training data in (0,1). Defaults to 0.8.
        pos_label (int, optional): label for the positive (anomaly) class. Defaults to -1.

    Returns:
        Dict: containing training parameters, fitted scaler, fitted OneHotEncoder, trained model, and performance metrics
    """
    utils.set_seeds()
    X = pd.read_csv(Path(config.DATA_DIR, "data.csv"))
    Y = X[[label_col]]
    X = X.drop(label_col, axis=1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data.get_data_splits(
        X, Y, train_size=train_size
    )
    X_train_processed, Y_train_processed, numerical_cols, ohe_enc = data.get_preprocessed_train(
        X_train, Y_train
    )
    scaler = RobustScaler()
    scaler.fit(X_train_processed)
    clf = OneClassSVM(
        gamma=args.gamma, shrinking=args.shrinking, verbose=args.verbose, max_iter=args.max_iter
    )
    clf.fit(scaler.transform(X_train_processed))
    del X_train_processed
    del Y_train_processed
    # Evaluation
    X_test_processed, Y_test_processed = data.get_test(X_test, ohe_enc, Y_test)
    Y_pred = clf.predict(scaler.transform(X_test_processed))
    metrics = precision_recall_fscore_support(
        Y_test_processed, Y_pred, average="binary", pos_label=pos_label
    )
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
    return {
        "args": args,
        "scaler": scaler,
        "ohe": ohe_enc,
        "model": clf,
        "performance": performance,
    }


def objective(args, trial):
    """This function defines the objective for hyperparameter tuning.

    Args:
        args (Namespace): training parameters
        trial (optuna.trial._trial.Trial): optimization trial

    Returns:
        float: The metric for hyperparameter tuning (f1).
    """
    args.gamma = trial.suggest_categorical("gamma", ["auto", "scale"])
    args.shrinking = trial.suggest_categorical("shrinking", [True, False])
    args.verbose = trial.suggest_categorical("verbose", [True, False])
    artifacts = train(args=args)
    performance = artifacts["performance"]
    return performance["f1"]
