import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into offline/online sets.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to diabetes_binary_health_indicators_BRFSS2015.csv",
    )
    parser.add_argument("--offline", default="offline.csv", help="Output offline csv")
    parser.add_argument("--online", default="online.csv", help="Output online csv")
    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column (default: auto-detect)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_label_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    if "Diabetes_binary" in df.columns:
        return "Diabetes_binary"
    if "Diabetes_012" in df.columns:
        return "Diabetes_012"
    return df.columns[-1]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    label_col = resolve_label_col(df, args.label_col)

    offline_df, online_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[label_col],
    )

    offline_df.to_csv(args.offline, index=False)
    online_df.to_csv(args.online, index=False)

    print(f"Saved offline split: {args.offline} ({len(offline_df)} rows)")
    print(f"Saved online split: {args.online} ({len(online_df)} rows)")
    print(f"Label column: {label_col}")


if __name__ == "__main__":
    main()
