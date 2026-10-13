# src/pipelines/training/train_xgboost_kmeans.py
"""
Train XGBoost (Spark) using K-means undersampling on the training partition.

- Uses `data/processed/transformed_data_2M.parquet` as input (2M records)
- Splits data as 60/20/20 (train/validation/test)
- Applies K-means to non-fraud records in the training partition only to create
  an undersampled training set with a target non-fraud percentage.
- Trains SparkXGBClassifier and evaluates on validation and test sets.
- Saves model to `models/best_xgb_model_kmeans_{percent}NF`.

Note: Requires the `ml.dmlc:xgboost4j-spark` package available to Spark.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# XGBoost Spark
from xgboost.spark import SparkXGBClassifier

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TARGET_NON_FRAUD_PERCENTAGE = 0.90  # keep 90% non-fraud in undersampled training set


def main():
    spark = SparkSession.builder \
        .appName(f"TrainXGB_KMeans_{int(TARGET_NON_FRAUD_PERCENTAGE*100)}NF") \
        .config("spark.driver.memory", "12g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data_2M.parquet'
    model_output_path = f'models/best_xgb_model_kmeans_{int(TARGET_NON_FRAUD_PERCENTAGE*100)}NF'

    logging.info("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # 60/20/20 split
    train_data, validation_data, test_data = df_transformed.randomSplit([0.6, 0.2, 0.2], seed=42)
    logging.info(f"Data split into Train ({train_data.count()} rows), Validation ({validation_data.count()} rows), Test ({test_data.count()} rows)")

    train_data.persist()
    validation_data.persist()
    test_data.persist()

    fraud_df = train_data.filter(col("isFraud") == 1)
    non_fraud_df = train_data.filter(col("isFraud") == 0)

    num_frauds = fraud_df.count()
    num_non_frauds = non_fraud_df.count()
    logging.info(f"Training set composition: {num_frauds} frauds, {num_non_frauds} non-frauds.")

    if num_frauds == 0 or TARGET_NON_FRAUD_PERCENTAGE >= 1.0 or TARGET_NON_FRAUD_PERCENTAGE <= 0.0:
        logging.error("Invalid target non-fraud percentage or zero frauds. Skipping K-means undersampling.")
        k_value = num_non_frauds
    else:
        target_fraud_percentage = 1.0 - TARGET_NON_FRAUD_PERCENTAGE
        k_value = int(num_frauds * (TARGET_NON_FRAUD_PERCENTAGE / target_fraud_percentage))

    k_value = min(k_value, num_non_frauds)
    logging.info(f"Applying K-means to non-fraud data with k={k_value} to target {TARGET_NON_FRAUD_PERCENTAGE*100:.0f}% non-fraud in train set.")

    kmeans = KMeans(featuresCol="features", k=k_value, seed=42)
    kmeans_model = kmeans.fit(non_fraud_df)

    cluster_centers = kmeans_model.clusterCenters()
    undersampled_non_fraud_data = [(Vectors.dense(center), 0) for center in cluster_centers]
    undersampled_non_fraud_df = spark.createDataFrame(undersampled_non_fraud_data, ["features", "isFraud"]) 
    logging.info(f"Created undersampled non-fraud DataFrame with {undersampled_non_fraud_df.count()} rows.")

    balanced_train_df = undersampled_non_fraud_df.union(fraud_df.select("features", "isFraud"))
    logging.info(f"Final balanced training set size: {balanced_train_df.count()} rows.")
    balanced_train_df.groupBy("isFraud").count().show()

    balanced_train_df.persist()
    train_data.unpersist()

    # Train XGBoost (Spark)
    logging.info("Training XGBoost (Spark) on K-means balanced data...")
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="isFraud",
        max_depth=7,
        n_estimators=100,
        seed=42
    )

    xgb_model = xgb.fit(balanced_train_df)

    balanced_train_df.unpersist()

    # Evaluate on validation set first
    logging.info("Evaluating on validation set...")
    val_predictions = xgb_model.transform(validation_data)
    val_predictions.persist()

    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    val_auc = evaluator_auc.evaluate(val_predictions)

    tp = val_predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = val_predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = val_predictions.filter("prediction = 0.0 AND isFraud = 1").count()

    val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

    logging.info(f"Validation AUC: {val_auc:.4f}")
    logging.info(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    val_predictions.unpersist()

    # Final evaluation on test set
    logging.info("Starting final evaluation on the original unseen (and unbalanced) test set...")
    test_predictions = xgb_model.transform(test_data)
    test_predictions.persist()

    test_auc = evaluator_auc.evaluate(test_predictions)
    tp = test_predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = test_predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = test_predictions.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = test_predictions.filter("prediction = 0.0 AND isFraud = 0").count()

    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    logging.info("\n--- FINAL MODEL REPORT (XGBoost trained with K-means Undersampling) ---")
    logging.info(f"Target Non-Fraud Ratio during K-means: {TARGET_NON_FRAUD_PERCENTAGE*100:.0f}%")
    logging.info("Confusion Matrix (Test):")
    test_predictions.groupBy("isFraud", "prediction").count().show()
    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")

    logging.info("\nPerformance Metrics (on unbalanced Test Data):")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC:      {test_auc:.4f}")
    logging.info(f"Precision (fraud): {test_precision:.4f}")
    logging.info(f"Recall (fraud):    {test_recall:.4f}")
    logging.info(f"F1-Score (fraud):  {test_f1:.4f}")

    test_predictions.unpersist()
    validation_data.unpersist()
    test_data.unpersist()

    xgb_model.write().overwrite().save(model_output_path)
    logging.info(f"XGBoost model (trained with K-means) saved to '{model_output_path}'")

    spark.stop()


if __name__ == '__main__':
    main()
