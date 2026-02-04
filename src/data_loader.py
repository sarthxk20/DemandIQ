from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    """
    Loads retail sales data for DemandIQ.

    Expected file:
    data/raw/train.csv

    The dataset is intentionally not included in the repository
    due to size and licensing constraints.
    """

    data_path = Path(data_path)
    train_path = data_path / "train.csv"

    # Graceful failure if data is missing
    if not train_path.exists():
        st.error(
            "❌ Dataset not found.\n\n"
            "Expected file location:\n"
            "`data/raw/train.csv`\n\n"
            "The dataset is not included in this repository.\n"
            "Please see the README for instructions on running the app locally."
        )
        st.stop()

    # Robust CSV loading for messy real-world data
    try:
        train = pd.read_csv(
            train_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        st.error(
            "❌ Failed to load dataset.\n\n"
            f"Error details:\n{e}"
        )
        st.stop()

    return train, None
