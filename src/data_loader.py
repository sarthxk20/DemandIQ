from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    data_path = Path(data_path)
    train_path = data_path / "train.csv"

    if not train_path.exists():
        st.warning(
            "⚠️ Dataset not found.\n\n"
            "The raw dataset is not included in this public repository.\n\n"
            "To run locally:\n"
            "1. Place `train.csv` in `data/raw/`\n"
            "2. Restart the app\n\n"
            "See README for details."
        )
        st.stop()   # ⛔ HARD STOP — NOTHING after this runs

    train = pd.read_csv(
        train_path,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    return train
