import argparse
import csv
import json
import time
from pathlib import Path

from kafka import KafkaProducer

from feature_transformations import resolve_label_col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send data rows to Kafka as JSON.")
    parser.add_argument("--input", default="offline.csv", help="Input CSV file")
    parser.add_argument("--topic", default="health_data", help="Kafka topic")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between records")
    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column (default: auto-detect)",
    )
    return parser.parse_args()


def normalize_row(row: dict, label_col: str) -> dict:
    row = dict(row)
    if label_col in row:
        row.pop(label_col)
    normalized = {}
    for key, value in row.items():
        if value == "":
            normalized[key] = None
        else:
            try:
                normalized[key] = float(value)
            except ValueError:
                normalized[key] = value
    return normalized


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        label_col = resolve_label_col(reader.fieldnames, args.label_col)

        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        sent = 0
        for row in reader:
            payload = normalize_row(row, label_col)
            producer.send(args.topic, payload)
            sent += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        producer.flush()
        print(f"Sent {sent} records to topic '{args.topic}'.")


if __name__ == "__main__":
    main()
