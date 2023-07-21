import mlflow
from typer.testing import CliRunner
from IDSAnoShift import main
from IDSAnoShift.main import app
import pytest
from pathlib import Path
from config import config

runner = CliRunner()
args_fp = Path(config.BASE_DIR, "tests", "code", "test_args.json")
def delete_experiment(exp_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(exp_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)

def test_extract_data():
    result = runner.invoke(app, ["extract-data"])
    assert result.exit_code == 0

@pytest.mark.training
def test_train_model():
    exp_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke(app, ["train-model",
                                 f"{args_fp}",
                                 f"{exp_name}",
                                 f"{run_name}"
                                 ])
    assert result.exit_code == 0
    delete_experiment(exp_name)

@pytest.mark.training
def test_optimize():
    study_name = "test_opt_study"
    num_trials = 1
    result = runner.invoke(app, ["optimize",
                                 f"{args_fp}",
                                 f"{study_name}",
                                 f"{num_trials}"
                                 ])
    assert result.exit_code == 0
    delete_experiment(study_name)

def test_load_artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id)
    assert len(artifacts)

def test_predict_label():
    traffic_data = ["c041","other","c263","c363","0","0.0","0.0","0.41","0","0","0.0","0.0","0.0","SF"]
    prediction = main.predict_label(traffic_data)
    assert prediction[0]["predicted_tags"]