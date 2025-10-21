from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_lr_thresholds():
    spark = SparkSession.builder \
        .appName("ThresholdEvaluationLR") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    # --- CHANGE 1: Point to the tuned LR model ---
    model_path = 'models/best_lr_model_tuned'

    print(f"Loading preprocessed data and the tuned LR model from '{model_path}'...")
    df_transformed = spark.read.parquet(transformed_data_path)
    try:
        # --- CHANGE 2: Load the correct model class ---
        model = LogisticRegressionModel.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{model_path}'.")
        # --- CHANGE 3: Mention the correct training script ---
        print("Certifique-se de que você treinou e salvou este modelo primeiro (usando '4_train_lr.py').")
        spark.stop()
        return

    # Use the same split as the training script for consistency
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)

    print("Making predictions to get probabilities...")
    predictions = model.transform(test_data)

    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    predictions.persist()

    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    print("\n" + "="*50)
    print(f"Área Sob a Curva ROC (AUC): {auc:.4f}")
    print("="*50)

    # --- CHANGE 4: Adjust thresholds to include LOWER values ---
    # Since LR is conservative, the optimal threshold might be much lower than 0.5
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # Added lower thresholds

    print("\n--- Evaluating Performance at Different Thresholds ---")
    best_f1 = -1
    best_threshold = -1
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold:.2f} ---") # Formatted threshold

        preds_with_threshold = predictions.withColumn("prediction_adj",
            when(col("fraud_prob") >= threshold, 1.0).otherwise(0.0)
        )

        tp = preds_with_threshold.filter("prediction_adj = 1.0 AND isFraud = 1").count()
        fp = preds_with_threshold.filter("prediction_adj = 1.0 AND isFraud = 0").count()
        fn = preds_with_threshold.filter("prediction_adj = 0.0 AND isFraud = 1").count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"(TP: {tp}, FP: {fp}, FN: {fn})")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

        # Keep track of the best threshold based on F1-Score
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

    print("\n" + "="*50)
    print(f"🏆 Best Threshold found for Logistic Regression: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    print("="*50)

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    evaluate_lr_thresholds() # Renamed function call