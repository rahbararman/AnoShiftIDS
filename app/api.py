import json
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from numpyencoder import NumpyEncoder

from app.schemas import PredictPayload
from config import config
from config.config import logger
from IDSAnoShift import main, predict, utils

app = FastAPI(
    title="IDS Ano Shift ",
    description="Identify anomalous behavior in network traffic.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id)
    logger.info("Ready for inference!")


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request):
    """Get the performance metrics."""
    performance = artifacts["performance"]
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": performance,
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload):
    """Predict labels for traffic data."""
    records = [t.record for t in payload.records]
    col_names = utils.load_dict(Path(config.CONFIG_DIR, "col_names.json"))["names"]
    traffic_data = pd.DataFrame(np.array(records), columns=col_names)
    predictions = predict.predict(traffic_data=traffic_data, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": json.dumps(predictions, indent=2, cls=NumpyEncoder)},
    }
    return response
