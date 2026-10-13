# src/pipelines/training/train_xgb_weighted.py
"""
Train XGBoost (Spark) using class weighting on the training partition.

Note: SparkXGBClassifier may accept `weight_col` parameter name. We pass the
training DataFrame with a `weight` column and set the classifier's weight_col.
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from xgboost.spark import SparkXGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRANSFORMED_PATH = 'data/processed/transformed_data_2M.parquet'
MODEL_OUTPUT_PATH = 'models/best_xgb_model_weighted'


def main():
    spark = SparkSession.builder \
        .appName("TrainXGB_Weighted") \
        .config("spark.driver.memory", "12g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    logging.info("Loading data...")
    df = spark.read.parquet(TRANSFORMED_PATH)

    train_data, validation_data, test_data = df.randomSplit([0.6, 0.2, 0.2], seed=42)
    logging.info(f"Splits: Train={train_data.count()}, Val={validation_data.count()}, Test={test_data.count()}")

    train_data.persist()
    validation_data.persist()
    test_data.persist()

    num_frauds = train_data.filter(col('isFraud') == 1).count()
    num_nonfrauds = train_data.filter(col('isFraud') == 0).count()
    logging.info(f"Train counts: frauds={num_frauds}, nonfrauds={num_nonfrauds}")

    if num_frauds == 0:
        logging.error("No frauds in training set; aborting weight-based training.")
        return

    balance_ratio = float(num_nonfrauds) / float(num_frauds)
    logging.info(f"Computed balance ratio (nonfraud/fraud) = {balance_ratio:.2f}")

    train_weighted = train_data.withColumn('weight', when(col('isFraud') == 1, balance_ratio).otherwise(1.0))
    logging.info("Added 'weight' column to training set.")

    # Train Spark XGBoost - pass weight_col parameter
    logging.info("Training XGBoost (Spark) with weight column...")
    xgb = SparkXGBClassifier(features_col='features', label_col='isFraud', weight_col='weight', max_depth=7, n_estimators=100, seed=42)
    xgb_model = xgb.fit(train_weighted)

    evaluator = BinaryClassificationEvaluator(labelCol='isFraud')

    logging.info("Evaluating on validation set...")
    val_preds = xgb_model.transform(validation_data)
    val_auc = evaluator.evaluate(val_preds)
    logging.info(f"Validation AUC: {val_auc:.4f}")

    logging.info("Evaluating on test set...")
    test_preds = xgb_model.transform(test_data)
    test_preds.persist()

    tp = test_preds.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = test_preds.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = test_preds.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = test_preds.filter("prediction = 0.0 AND isFraud = 0").count()

    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    test_auc = evaluator.evaluate(test_preds)

    logging.info(f"Test AUC: {test_auc:.4f}")
    logging.info(f"Test (TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    logging.info(f"Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    xgb_model.write().overwrite().save(MODEL_OUTPUT_PATH)
    logging.info(f"Saved weighted XGBoost model to {MODEL_OUTPUT_PATH}")

    test_preds.unpersist()
    validation_data.unpersist()
    test_data.unpersist()
    spark.stop()


if __name__ == '__main__':
    main()
