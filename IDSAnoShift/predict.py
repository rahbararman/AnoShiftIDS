from IDSAnoShift import data


def predict(traffic_data, artifacts):
    """This functions predicts the labels for given traffic data.

    Args:
        traffic_data (pd.DataFrame): DataFrame containing raw features for traffic
        artifacts (Dict): artifacts for trained model

    Returns:
        Dict: dictionary containing the input features and the predicted labels
    """
    traffic_data_processed, _ = data.get_test(traffic_data, artifacts["ohe"])
    Y_pred = artifacts["model"].predict(artifacts["scaler"].transform(traffic_data_processed))
    index_to_class = {-1: "anomaly", 1: "normal"}
    predictions = [
        {
            "input_traffic": list(traffic_data.iloc[i, :]),
            "predicted_tags": index_to_class[Y_pred[i]],
        }
        for i in range(len(Y_pred))
    ]
    return predictions
