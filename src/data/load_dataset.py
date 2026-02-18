from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import RAW_PARQUET_PATH

@dataclass(frozen=True)
class DataLoadResult:
    df: pd.DataFrame
    path: Path

def load_parquet(path: str | Path) -> DataLoadResult:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p.resolve()}")

    df = pd.read_parquet(p, engine="pyarrow")

    # basic smoke checks (we'll formalize later in Step 2)
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns detected.")

    return DataLoadResult(df=df, path=p)


def main() -> None:
    result = load_parquet(RAW_PARQUET_PATH)
    df = result.df

    print("✅ Loaded Parquet")
    print(f"Path: {result.path.resolve()}")
    print(f"Shape: {df.shape}")
    print("Columns (first 30):", list(df.columns[:30]))
    print("\nDtypes (top 20):")
    print(df.dtypes.head(20))
    print("\nSample:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

