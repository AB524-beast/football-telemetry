from pathlib import Path
import pandas as pd

for f in sorted(Path("data/raw").rglob("*.csv")):
    df = pd.read_csv(f, nrows=2000)
    print(f"\n=== {f} ===  shape(sample)={df.shape}")
    print("columns:", list(df.columns))
    print(df.head(3).to_string())