import argparse
import os
from typing import List

from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, struct, to_json
from pyspark.sql.types import DoubleType, StructField, StructType

from feature_transformations import extract_feature_cols_from_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online Spark streaming for predictions.")
    parser.add_argument(
        "--model-dir",
        default="models/best_model",
        help="Path to saved model",
    )
    parser.add_argument("--input-topic", default="health_data", help="Kafka input topic")
    parser.add_argument(
        "--output-topic",
        default="health_data_predicted",
        help="Kafka output topic",
    )
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap")
    return parser.parse_args()


def build_schema(feature_cols: List[str]) -> StructType:
    return StructType([StructField(col_name, DoubleType(), True) for col_name in feature_cols])


def build_spark(app_name: str) -> SparkSession:
    project_root = Path(__file__).resolve().parent
    hadoop_home = project_root / "hadoop"
    os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
    os.environ.setdefault("hadoop.home.dir", str(hadoop_home))
    os.environ.setdefault("HADOOP_DISABLE_IO_NATIVE", "1")
    os.environ["PATH"] = f"{hadoop_home / 'bin'};{os.environ.get('PATH', '')}"
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )


def main() -> None:
    args = parse_args()
    spark = build_spark("online-streaming")
    spark.sparkContext.setLogLevel("WARN")

    model_path = args.model_dir
    if not model_path.startswith("file:"):
        model_path = Path(model_path).resolve().as_uri()
    model = PipelineModel.load(model_path)
    feature_cols = extract_feature_cols_from_model(model)
    input_schema = build_schema(feature_cols)

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribe", args.input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        kafka_df.select(from_json(col("value").cast("string"), input_schema).alias("data"))
        .select("data.*")
    )

    predictions = model.transform(parsed).withColumn(
        "prediction", col("prediction").cast("int")
    )

    output_cols = [col(c) for c in feature_cols] + [col("prediction")]
    output = predictions.select(to_json(struct(*output_cols)).alias("value"))

    query = (
        output.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("topic", args.output_topic)
        .option("checkpointLocation", "checkpoints/health_data_predicted")
        .outputMode("append")
        .start()
    )

    print(f"Consuming from: {args.input_topic}")
    print(f"Producing to: {args.output_topic}")
    query.awaitTermination()


if __name__ == "__main__":
    main()
