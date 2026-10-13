# Convert Parquet files to CSV
# Requires: pandas, pyarrow (il motore con cui pandas legge i file Parquet)
import argparse
from pathlib import Path

import pandas as pd


def convert_parquet_to_csv(parquet_file, csv_file):
    df = pd.read_parquet(parquet_file)
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Convert Parquet to CSV')
    parser.add_argument('parquet_file', type=str, help='Path to the input Parquet file')
    parser.add_argument('csv_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()

    df = convert_parquet_to_csv(args.parquet_file, args.csv_file)
    print(f"Converted {args.parquet_file} -> {args.csv_file} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
