import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import DoubleType
from imblearn.over_sampling import ADASYN

def main():
    spark = SparkSession.builder \
        .appName("TrainRandomForest_ADASYN") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.local.ip", "192.168.2.14") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_rf_model_adasyn'

    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path).repartition(2)

    # Verify test set distribution
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    test_count = test_data.count()
    fraud_count = test_data.filter(col("isFraud") == 1).count()
    print(f"Test set size: {test_count}, Fraud cases: {fraud_count}, Fraud ratio: {fraud_count / test_count:.6f}")

    # Convert training data to Pandas for ADASYN
    print("Converting training data to Pandas for ADASYN application...")
    train_pd = train_data.select("features", "isFraud").toPandas()
    X_train = pd.DataFrame(train_pd['features'].apply(lambda x: x.toArray()).tolist())
    y_train = train_pd['isFraud']

    # Apply ADASYN
    print("Applying ADASYN to balance the training data...")
    adasyn = ADASYN(sampling_strategy=0.2, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_resampled.shape}")
    print("Class distribution after ADASYN:")
    print(y_resampled.value_counts())

    # Convert balanced data back to Spark DataFrame
    print("Converting balanced data back to Spark DataFrame...")
    resampled_data = [(Vectors.dense(row.values), int(y_resampled[index])) for index, row in X_resampled.iterrows()]
    balanced_train_df = spark.createDataFrame(resampled_data, ["features", "isFraud"])

    # Train Random Forest
    print("Training Random Forest on ADASYN-balanced data...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=100, maxDepth=15, seed=42)
    rf_model = rf.fit(balanced_train_df)

    # Train GBT
    print("Training Gradient Boosted Trees on ADASYN-balanced data...")
    gbt = GBTClassifier(featuresCol="features", labelCol="isFraud", maxIter=50, seed=42)
    gbt_model = gbt.fit(balanced_train_df)

    # Evaluate on test set
    print("\n--- Final Evaluation on Unseen (and Unbalanced) Test Data ---")
    rf_predictions = rf_model.transform(test_data)
    gbt_predictions = gbt_model.transform(test_data).select("features", "isFraud", col("prediction").alias("gbt_prediction"), col("probability").alias("gbt_probability"))
    rf_predictions = rf_predictions.select("features", "isFraud", col("prediction").alias("rf_prediction"), col("probability").alias("rf_probability"))

    # Combine predictions for ensemble
    combined = rf_predictions.join(gbt_predictions, ["features", "isFraud"])
    combined = combined.withColumn("ensemble_prediction", when((col("rf_prediction") == 1.0) & (col("gbt_prediction") == 1.0), 1.0).otherwise(0.0))

    # Adjust threshold for Random Forest
    def extract_prob(v):
        return float(v[1])  # Probability of class 1
    extract_prob_udf = udf(extract_prob, DoubleType())
    rf_predictions = rf_predictions.withColumn("probability_score", extract_prob_udf(col("rf_probability")))

    # Test multiple thresholds
    thresholds = [0.5, 0.6, 0.65, 0.7]
    for threshold in thresholds:
        rf_predictions = rf_predictions.withColumn("adjusted_prediction", when(col("probability_score") >= threshold, 1.0).otherwise(0.0))
        tp = rf_predictions.filter("adjusted_prediction = 1.0 AND isFraud = 1").count()
        fp = rf_predictions.filter("adjusted_prediction = 1.0 AND isFraud = 0").count()
        fn = rf_predictions.filter("adjusted_prediction = 0.0 AND isFraud = 1").count()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\nRandom Forest - Threshold: {threshold}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    # Evaluate ensemble
    tp = combined.filter("ensemble_prediction = 1.0 AND isFraud = 1").count()
    fp = combined.filter("ensemble_prediction = 1.0 AND isFraud = 0").count()
    fn = combined.filter("ensemble_prediction = 0.0 AND isFraud = 1").count()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("\nEnsemble (RF + GBT) Results:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    # Save the best model
    rf_model.write().overwrite().save(model_output_path)
    print(f"\nBest Random Forest model saved to '{model_output_path}'")
    spark.stop()

if __name__ == "__main__":
    main()