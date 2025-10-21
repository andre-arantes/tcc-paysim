import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    spark = SparkSession.builder \
        .appName("TrainXGBoost") \
        .config("spark.driver.memory", "8g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data_2M.parquet'
    model_output_path = 'models/best_xgb_model_final'

    logging.info("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    logging.info(f"Data split into Training ({train_data.count()} rows) and Test ({test_data.count()} rows).")

    train_data_xgb = train_data.withColumnRenamed("isFraud", "label")
    test_data_xgb = test_data.withColumnRenamed("isFraud", "label")

    logging.info("Initializing XGBoost Classifier...")
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        max_depth=7,
        n_estimators=100,
        seed=42
    )
    
    logging.info("Training XGBoost model on the training data... This might take a while.")
    xgb_model = xgb.fit(train_data_xgb)

    logging.info("Starting initial evaluation on the unseen test set...")
    predictions = xgb_model.transform(test_data_xgb)
    predictions.persist()

    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)

    tp = predictions.filter("prediction = 1.0 AND label = 1").count()
    fp = predictions.filter("prediction = 1.0 AND label = 0").count()
    fn = predictions.filter("prediction = 0.0 AND label = 1").count()
    tn = predictions.filter("prediction = 0.0 AND label = 0").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    logging.info("\n--- INITIAL MODEL REPORT (on Test Data, Threshold=0.5) ---")
    logging.info("Confusion Matrix:")
    predictions.withColumnRenamed("label", "isFraud") \
               .groupBy("isFraud", "prediction").count().show()
    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    
    logging.info("\nPerformance Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC:      {auc:.4f}")
    logging.info(f"Precision (for fraud class): {precision:.4f}")
    logging.info(f"Recall (for fraud class):    {recall:.4f}")
    logging.info(f"F1-Score (for fraud class):  {f1_score:.4f}")

    predictions.unpersist()
    xgb_model.write().overwrite().save(model_output_path)
    logging.info(f"\n✅ XGBoost model successfully trained and saved to '{model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()