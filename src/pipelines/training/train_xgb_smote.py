# src/pipelines/training/train_xgb_smote.py
"""
Train XGBoost (Spark) using SMOTE applied to the training partition.

Requires: ml.dmlc:xgboost4j-spark
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans

from xgboost.spark import SparkXGBClassifier

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_OUTPUT_PATH = 'models/best_xgb_model_smote'
TRANSFORMED_PATH = 'data/processed/transformed_data_2M.parquet'


def apply_smote_on_driver(spark, minority_df, majority_count, target_ratio=0.5):
    pandas_minority = minority_df.toPandas()
    X = list(pandas_minority['features'].apply(lambda v: list(v)))
    y = list(pandas_minority['isFraud'].astype(int))

    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        res_rows = [(Vectors.dense(x), int(yv)) for x, yv in zip(X_res, y_res)]
        res_df = spark.createDataFrame(res_rows, ["features", "isFraud"])
        return res_df
    except Exception:
        logging.warning("imblearn not available - falling back to simple oversampling.")
        needed = int((majority_count * target_ratio) - len(X))
        if needed <= 0:
            return minority_df
        replicated = []
        for i in range(needed):
            row = pandas_minority.iloc[i % len(pandas_minority)]
            replicated.append((row['features'], int(row['isFraud'])))
        res_df = spark.createDataFrame(replicated, ["features", "isFraud"]) 
        combined = minority_df.union(res_df)
        return combined


def main():
    spark = SparkSession.builder.appName("TrainXGB_SMOTE")\
        .config("spark.driver.memory", "12g")\
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    logging.info("Loading preprocessed data...")
    df = spark.read.parquet(TRANSFORMED_PATH)

    train_data, validation_data, test_data = df.randomSplit([0.6, 0.2, 0.2], seed=42)
    logging.info(f"Splits: Train={train_data.count()}, Val={validation_data.count()}, Test={test_data.count()}")

    train_data.persist()
    validation_data.persist()
    test_data.persist()

    fraud_df = train_data.filter(col("isFraud") == 1)
    non_fraud_df = train_data.filter(col("isFraud") == 0)

    num_frauds = fraud_df.count()
    num_non_frauds = non_fraud_df.count()
    logging.info(f"Train composition before SMOTE: frauds={num_frauds}, nonfrauds={num_non_frauds}")

    minority_resampled = apply_smote_on_driver(spark, fraud_df.select("features", "isFraud"), num_non_frauds, target_ratio=1.0)
    balanced_train = non_fraud_df.select("features", "isFraud").union(minority_resampled)

    logging.info(f"Balanced training set size: {balanced_train.count()} rows")
    balanced_train.groupBy('isFraud').count().show()

    balanced_train.persist()
    train_data.unpersist()

    logging.info("Training XGBoost (Spark) with SMOTE-balanced data...")
    xgb = SparkXGBClassifier(features_col="features", label_col="isFraud", max_depth=7, n_estimators=100, seed=42)
    xgb_model = xgb.fit(balanced_train)

    balanced_train.unpersist()

    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")

    logging.info("Evaluating on validation set...")
    val_preds = xgb_model.transform(validation_data)
    val_auc = evaluator.evaluate(val_preds)
    logging.info(f"Validation AUC: {val_auc:.4f}")

    logging.info("Evaluating on test set...")
    test_preds = xgb_model.transform(test_data)
    test_preds.persist()
    test_auc = evaluator.evaluate(test_preds)

    tp = test_preds.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = test_preds.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = test_preds.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = test_preds.filter("prediction = 0.0 AND isFraud = 0").count()

    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

    logging.info(f"Test AUC: {test_auc:.4f}")
    logging.info(f"Test (TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    logging.info(f"Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    xgb_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    logging.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    validation_data.unpersist()
    test_data.unpersist()
    spark.stop()


if __name__ == '__main__':
    main()
