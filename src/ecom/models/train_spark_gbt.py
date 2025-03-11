"""Spark MLlib gradient-boosted trees: the distributed counterpart to XGBoost.

Trains on the same Hive ``model_dataset`` inside Spark so it scales to the full data without
collecting it to the driver; used as a scale comparison against the single-node models.
"""

from __future__ import annotations

import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ecom.config import Config
from ecom.layout import table_path
from ecom.transform.build_features import NUMERIC_FEATURES


def train_spark_gbt(spark: SparkSession, cfg: Config) -> dict:
    df = spark.read.parquet(str(table_path(cfg, "model_dataset"))).fillna(0.0, NUMERIC_FEATURES)
    train = df.where(F.col("split") == "train")
    valid = df.where(F.col("split") == "valid")
    test = df.where(F.col("split") == "test")
    pipeline = Pipeline(
        stages=[
            StringIndexer(
                inputCol="top_category", outputCol="top_category_idx", handleInvalid="keep"
            ),
            OneHotEncoder(inputCols=["top_category_idx"], outputCols=["top_category_ohe"]),
            VectorAssembler(
                inputCols=[*NUMERIC_FEATURES, "top_category_ohe"], outputCol="features"
            ),
            GBTClassifier(
                labelCol="label",
                featuresCol="features",
                maxIter=int(cfg.get("ml.spark_gbt_max_iter", 40)),
                maxDepth=5,
                stepSize=0.1,
                subsamplingRate=0.8,
                seed=cfg.seed,
            ),
        ]
    )
    t0 = time.perf_counter()
    model = pipeline.fit(train)
    train_s = time.perf_counter() - t0
    pred = model.transform(test).withColumn("p", vector_to_array("probability")[1])
    metrics = {
        name: BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName=metric
        ).evaluate(pred)
        for name, metric in (("roc_auc", "areaUnderROC"), ("pr_auc", "areaUnderPR"))
    }
    scored = pred.select("label", "p").toPandas()
    scored_valid = (
        model.transform(valid)
        .withColumn("p", vector_to_array("probability")[1])
        .select("label", "p")
        .toPandas()
    )
    return {
        "metrics_spark": metrics,
        "y": scored["label"].to_numpy(),
        "p": scored["p"].to_numpy(),
        "y_valid": scored_valid["label"].to_numpy(),
        "p_valid": scored_valid["p"].to_numpy(),
        "train_seconds": train_s,
        "train_rows": train.count(),
        "params": {
            "maxIter": int(cfg.get("ml.spark_gbt_max_iter", 40)),
            "maxDepth": 5,
            "stepSize": 0.1,
            "subsamplingRate": 0.8,
        },
    }
