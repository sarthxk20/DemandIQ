from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    data_path = Path(data_path)
    parquet_path = data_path / "train.parquet"

    if not parquet_path.exists():
        st.error(
            "❌ Dataset not found.\n\n"
            "Expected file: data/raw/train.parquet\n\n"
            "Please ensure the dataset is available in the repository."
        )
        st.stop()

    return pd.read_parquet(parquet_path)
