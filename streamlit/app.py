from pathlib import Path

import pandas as pd

import streamlit as st
from config import config
from IDSAnoShift import main, utils

st.title("IDS with AnoShift")

st.header("Data")

data_fp = Path(config.DATA_DIR, "data.csv")
df = pd.read_csv(data_fp)
st.text(f"Traffic data (number of records: {len(df)})")
st.write(df)

st.header("Performance")

performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.write(performance)

st.header("Inference")

features = st.text_input(
    "Enter traffic features:", "c015,other,c20,c30,4,1.0,1.0,0.8,90,100,0.0,1.0,1.0,S0"
)
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_label(traffic_data=features.split(","), run_id=run_id)
st.write(prediction)
