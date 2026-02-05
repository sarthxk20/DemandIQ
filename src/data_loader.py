from pathlib import Path
import pandas as pd

def load_data(data_path="data/raw"):
    parquet_path = Path(data_path) / "train.parquet"
    return pd.read_parquet(parquet_path)
