from pathlib import Path
import pandas as pd
import streamlit as st


def load_data(data_path="data/raw"):
    """
    Loads and normalizes retail sales data for DemandIQ.

    Expected logical columns (after normalization):
    - Date
    - Store
    - Sales
    """

    data_path = Path(data_path)
    train_path = data_path / "train.csv"

    if not train_path.exists():
        st.error(
            "❌ Dataset not found.\n\n"
            "Expected file location:\n"
            "`data/raw/train.csv`\n\n"
            "The dataset is not included in this repository.\n"
            "Please see the README for instructions on running the app locally."
        )
        st.stop()

    # Load CSV defensively
    try:
        train = pd.read_csv(
            train_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        st.error(f"❌ Failed to load dataset.\n\n{e}")
        st.stop()

    # ---------------------------
    # Normalize column names
    # ---------------------------
    train.columns = (
        train.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Column mapping (raw -> expected)
    column_map = {
        "date": "date",
        "order_date": "date",
        "sales_date": "date",

        "store": "store",
        "store_id": "store",
        "outlet": "store",

        "sales": "sales",
        "revenue": "sales",
        "units_sold": "sales"
    }

    # Apply mapping
    train = train.rename(columns=column_map)

    # ---------------------------
    # Validate required columns
    # ---------------------------
    required_columns = {"date", "store", "sales"}
    missing = required_columns - set(train.columns)

    if missing:
        st.error(
            "❌ Dataset schema mismatch.\n\n"
            f"Missing required columns: {', '.join(missing)}\n\n"
            "Expected logical columns:\n"
            "- Date\n"
            "- Store\n"
            "- Sales"
        )
        st.stop()

    # Final formatting
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train = train.dropna(subset=["date", "sales", "store"])

    train = train.rename(columns={
        "date": "Date",
        "store": "Store",
        "sales": "Sales"
    })

    return train, None
