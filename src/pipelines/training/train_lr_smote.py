# src/pipelines/training/train_lr_smote.py
"""
Train Logistic Regression using SMOTE applied to the training partition.

Approach:
- Uses `data/processed/transformed_data_2M.parquet` as input (2M records)
- Splits data as 60/20/20 (train/validation/test)
- Applies SMOTE to the training partition. If `imblearn` is available it will be
  used; otherwise falls back to simple oversampling with replacement.
- Trains LogisticRegression and evaluates on validation and test sets.
- Saves model to `models/best_lr_model_smote`.

Note: This script collects the minority class to the driver for SMOTE synthesis,
which is acceptable for moderate minority sizes (typical in PaySim). If your
minority set is too large, consider distributed SMOTE implementations.
"""

import logging
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_OUTPUT_PATH = 'models/best_lr_model_smote'
TRANSFORMED_PATH = 'data/processed/transformed_data_2M.parquet'


def apply_smote_on_driver(spark, minority_df, majority_count, target_ratio=0.5):
    """Collect minority class to driver, run SMOTE (imblearn) if available, else naive oversample."""
    pandas_minority = minority_df.toPandas()
    X = list(pandas_minority['features'].apply(lambda v: list(v)))
    y = list(pandas_minority['isFraud'].astype(int))

    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        # Convert back to spark DataFrame
        res_rows = [(Vectors.dense(x), int(yv)) for x, yv in zip(X_res, y_res)]
        res_df = spark.createDataFrame(res_rows, ["features", "isFraud"])
        return res_df
    except Exception:
        logging.warning("imblearn not available or SMOTE failed - falling back to simple oversampling.")
        # simple oversampling: sample minority with replacement to reach majority_count * target_ratio
        needed = int((majority_count * target_ratio) - len(X))
        if needed <= 0:
            return minority_df
        # replicate rows
        replicated = []
        for i in range(needed):
            row = pandas_minority.iloc[i % len(pandas_minority)]
            replicated.append((row['features'], int(row['isFraud'])))
        res_df = spark.createDataFrame(replicated, ["features", "isFraud"]) 
        combined = minority_df.union(res_df)
        return combined


def main():
    spark = SparkSession.builder.appName("TrainLR_SMOTE")\
        .config("spark.driver.memory", "10g")\
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

    # Goal: after SMOTE, target minority/majority ratio approx 1:1 (can be changed)
    try:
        # Collect minority and apply SMOTE on driver
        from pyspark.ml.linalg import Vectors
        minority_resampled = apply_smote_on_driver(spark, fraud_df.select("features", "isFraud"), num_non_frauds, target_ratio=1.0)
        balanced_train = non_fraud_df.select("features", "isFraud").union(minority_resampled)
    except Exception as e:
        logging.error(f"SMOTE resampling failed: {e}")
        balanced_train = train_data.select("features", "isFraud")

    logging.info(f"Balanced training set size: {balanced_train.count()} rows")
    balanced_train.groupBy('isFraud').count().show()

    balanced_train.persist()
    train_data.unpersist()

    # Train Logistic Regression
    logging.info("Training Logistic Regression...")
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")
    lr_model = lr.fit(balanced_train)

    balanced_train.unpersist()

    # Evaluate
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")

    logging.info("Evaluating on validation set...")
    val_preds = lr_model.transform(validation_data)
    val_preds.persist()
    val_auc = evaluator.evaluate(val_preds)

    # Confusion matrix counts
    vp_tp = val_preds.filter("prediction = 1.0 AND isFraud = 1").count()
    vp_fp = val_preds.filter("prediction = 1.0 AND isFraud = 0").count()
    vp_fn = val_preds.filter("prediction = 0.0 AND isFraud = 1").count()
    vp_tn = val_preds.filter("prediction = 0.0 AND isFraud = 0").count()

    val_precision = vp_tp / (vp_tp + vp_fp) if (vp_tp + vp_fp) > 0 else 0
    val_recall = vp_tp / (vp_tp + vp_fn) if (vp_tp + vp_fn) > 0 else 0
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

    logging.info(f"Validation AUC: {val_auc:.4f}")
    logging.info(f"Validation (TP: {vp_tp}, FP: {vp_fp}, FN: {vp_fn}, TN: {vp_tn})")
    logging.info(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    logging.info("Evaluating on test set...")
    test_preds = lr_model.transform(test_data)
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

    lr_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    logging.info(f"Model saved to {MODEL_OUTPUT_PATH}")

    validation_data.unpersist()
    test_data.unpersist()
    spark.stop()


if __name__ == '__main__':
    main()
