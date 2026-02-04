from typing import List, Tuple

from pyspark.ml.feature import VectorAssembler


def resolve_label_col(columns: List[str], preferred: str | None = None) -> str:
    if preferred and preferred in columns:
        return preferred
    if "Diabetes_binary" in columns:
        return "Diabetes_binary"
    if "Diabetes_012" in columns:
        return "Diabetes_012"
    return columns[-1]


def get_feature_columns(columns: List[str], label_col: str) -> List[str]:
    return [col for col in columns if col != label_col]


def build_preprocessing_stages(feature_cols: List[str], output_col: str = "features"):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=output_col,
        handleInvalid="keep",
    )
    return [assembler]


def extract_feature_cols_from_model(pipeline_model) -> List[str]:
    for stage in pipeline_model.stages:
        if isinstance(stage, VectorAssembler):
            return list(stage.getInputCols())
    raise ValueError("VectorAssembler stage not found in pipeline model.")


def resolve_label_and_features(
    columns: List[str], preferred_label: str | None = None
) -> Tuple[str, List[str]]:
    label_col = resolve_label_col(columns, preferred_label)
    feature_cols = get_feature_columns(columns, label_col)
    return label_col, feature_cols
