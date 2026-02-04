import argparse
import json

from kafka import KafkaConsumer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read predictions from Kafka.")
    parser.add_argument("--topic", default="health_data_predicted", help="Kafka topic")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    count = 0
    try:
        for message in consumer:
            count += 1
            print(message.value)
    except KeyboardInterrupt:
        print(f"Stopped. Total records received: {count}")


if __name__ == "__main__":
    main()
