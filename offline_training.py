import argparse
import json
import os
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from feature_transformations import build_preprocessing_stages, resolve_label_and_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline training for diabetes prediction.")
    parser.add_argument("--data", default="offline.csv", help="Offline training data CSV")
    parser.add_argument(
        "--model-dir",
        default="models/best_model",
        help="Path to save best model",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main() -> None:
    args = parse_args()
    spark = build_spark("offline-training")
    spark.sparkContext.setLogLevel("WARN")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Offline data not found: {data_path}")

    df = spark.read.csv(str(data_path), header=True, inferSchema=True)
    label_col, feature_cols = resolve_label_and_features(df.columns, args.label_col)

    # Convert multiclass labels (0, 1, 2) to binary (0 for no diabetes, 1 for diabetes/prediabetes)
    df = df.withColumn(label_col, F.when(F.col(label_col) > 0, 1.0).otherwise(0.0))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)

    preprocessing_stages = build_preprocessing_stages(feature_cols)
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    )

    lr = LogisticRegression(labelCol=label_col, featuresCol="features", maxIter=50)
    rf = RandomForestClassifier(
        labelCol=label_col, featuresCol="features", seed=args.seed
    )
    gbt = GBTClassifier(
        labelCol=label_col, featuresCol="features", seed=args.seed, maxIter=50
    )

    candidates = [
        (
            "logistic_regression",
            lr,
            ParamGridBuilder()
            .addGrid(lr.regParam, [0.0, 0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
            .build(),
        ),
        (
            "random_forest",
            rf,
            ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [5, 10])
            .build(),
        ),
        (
            "gbt",
            gbt,
            ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3, 5])
            .addGrid(gbt.stepSize, [0.05, 0.1])
            .build(),
        ),
    ]

    metrics = {}
    best = {"name": None, "model": None, "f1": -1.0}

    for name, estimator, grid in candidates:
        pipeline = Pipeline(stages=preprocessing_stages + [estimator])
        tvs = TrainValidationSplit(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            trainRatio=0.8,
            seed=args.seed,
        )

        tvs_model = tvs.fit(train_df)
        predictions = tvs_model.transform(test_df)
        f1 = evaluator.evaluate(predictions)
        metrics[name] = {"f1": float(f1)}

        if f1 > best["f1"]:
            best = {"name": name, "model": tvs_model.bestModel, "f1": f1}

        print(f"{name} F1 on test: {f1:.4f}")

    model_dir = Path(args.model_dir).resolve()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir.as_uri()
    best["model"].write().overwrite().save(model_save_path)

    metrics_path = model_dir.parent / "metrics.json"
    metrics["best_model"] = {"name": best["name"], "f1": float(best["f1"])}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved best model: {best['name']} (F1={best['f1']:.4f})")
    print(f"Model path: {model_dir}")
    print(f"Metrics saved to: {metrics_path}")

    spark.stop()


if __name__ == "__main__":
    main()
