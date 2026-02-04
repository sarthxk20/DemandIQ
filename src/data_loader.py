from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    """
    Loads retail sales data for DemandIQ.

    The raw dataset is intentionally not included in the public repository
    due to size and licensing considerations.
    """

    data_path = Path(data_path)
    train_path = data_path / "train.csv"

    if not train_path.exists():
        st.warning(
            "⚠️ Dataset not found.\n\n"
            "The raw dataset is not included in this public repository.\n\n"
            "To run this app locally:\n"
            "1. Place your dataset at `data/raw/train.csv`\n"
            "2. Restart the application\n\n"
            "See the README for more details."
        )
        st.stop()

    train = pd.read_csv(
        train_path,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    return train, None
