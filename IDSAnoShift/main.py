import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import List

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration import MLflowCallback

from config import config
from config.config import logger
from IDSAnoShift import predict, train, utils

app = typer.Typer()


@app.command()
def extract_data(label_col="18"):
    """This function imports the training data to the project directory.

    Args:
        label_col (str, optional): Name of the label column in the data. Defaults to '18'.
    """
    X = pd.read_csv(config.DATA_LOCATION)
    X.to_csv(Path(config.DATA_DIR, "data.csv"), index=False)
    logger.info("✅ Saved data!")


@app.command()
def train_model(args_fp, experiment_name, run_name, test_run=True):
    """This function trains a model given arguments.

    Args:
        args_fp (str): location of training parameters
        experiment_name (str): name of mlflow experiment
        run_name (str): name of mlflow run
    """
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": artifacts["performance"]["precision"]})
        mlflow.log_metrics({"recall": artifacts["performance"]["recall"]})
        mlflow.log_metrics({"f1": artifacts["performance"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["scaler"], Path(dp, "scaler.pkl"))
            joblib.dump(artifacts["ohe"], Path(dp, "ohe.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(artifacts["performance"], Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(args_fp, study_name, num_trials):
    """This function optimizes the objective and saves the best parameters.

    Args:
        args_fp (str): location of training parameters
        study_name (str): name of optuna study
        num_trials (int): number of trials in the hyperparameter optimization study
    """
    num_trials = int(num_trials)
    args = Namespace(**utils.load_dict(filepath=args_fp))
    study = optuna.create_study(study_name=study_name, direction="maximize")
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, trial), n_trials=num_trials, callbacks=[mlflow_callback]
    )
    print(f"Best value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)


def load_artifacts(run_id):
    """This function loads artifacts for a given run_id.

    Args:
        run_id (str): id of the run to load artifacts from

    Returns:
        Dict: artifacts for the run
    """
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    scaler = joblib.load(Path(artifacts_dir, "scaler.pkl"))
    ohe = joblib.load(Path(artifacts_dir, "ohe.pkl"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(Path(artifacts_dir, "performance.json"))

    return {"scaler": scaler, "ohe": ohe, "model": model, "performance": performance}


@app.command()
def predict_label(traffic_data: List[str], run_id=None):
    """This function predicts the label for given new data.

    Args:
        traffic_data (List): A list including traffic features
        run_id (str, optional): id of the run to load artifacts from. Defaults to None.

    Returns:
        Dict: dictionary containing the input features and the predicted label
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    col_names = utils.load_dict(Path(config.CONFIG_DIR, "col_names.json"))["names"]
    traffic_data = pd.DataFrame(np.array([traffic_data]), columns=col_names)
    prediction = predict.predict(traffic_data=traffic_data, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2, cls=NumpyEncoder))
    return prediction


if __name__ == "__main__":
    app()  # pragma: no cover, CLI app
