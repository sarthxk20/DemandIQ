from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    data_path = Path(data_path)

    train_path = data_path / "train.csv"

    if not train_path.exists():
        st.error(
            "❌ Data file not found.\n\n"
            "This application expects a `train.csv` file at:\n"
            "`data/raw/train.csv`\n\n"
            "The dataset is not included in the repository due to size/licensing reasons.\n"
            "Please follow the instructions in the README to run the app locally."
        )
        st.stop()

    try:
        train = pd.read_csv(train_path, encoding="utf-8")
    except UnicodeDecodeError:
        train = pd.read_csv(train_path, encoding="latin1")

    return train, None
