import pandas as pd
from pathlib import Path


def load_data(data_dir: str = "data/raw"):
    """
    Load and prepare Rossmann sales data.
    """
    data_path = Path(data_dir)

    train = pd.read_csv(train_path, encoding="latin1")
    store = pd.read_csv(data_path / "store.csv")

    # Parse date column correctly
    train["Date"] = pd.to_datetime(train["Date"])

    return train, store

