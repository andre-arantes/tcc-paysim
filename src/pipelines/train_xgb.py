# File: 5_train_xgb.py
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from xgboost.spark import SparkXGBClassifier

def main():
    spark = SparkSession.builder \
        .appName("TrainXGBoost") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_xgb_model_final'

    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    print("Starting training and validation for XGBOOST...")
    df_xgb = df_transformed.withColumnRenamed("isFraud", "label")
    xgb = SparkXGBClassifier(
        features_col="features", label_col="label", weight_col="weight",
        max_depth=7, n_estimators=100, seed=42
    )
    
    (train_data, test_data) = df_xgb.randomSplit([0.8, 0.2], seed=42)
    
    print("Training XGBoost model (80/20 split)...")
    xgb_model = xgb.fit(train_data)
    
    print("\n--- XGBoost Evaluation on Test Set ---")
    predictions = xgb_model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"XGBoost F1-Score on test set: {f1_score:.4f}")
    
    xgb_model.write().overwrite().save(model_output_path)
    print(f"\nBest XGBoost model saved to '{model_output_path}'")

    spark.stop()

if __name__ == "__main__":
    main()