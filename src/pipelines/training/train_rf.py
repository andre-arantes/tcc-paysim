# src/pipelines/3_train_rf.py (Enhanced with Logging and Metrics)
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.classification import RandomForestClassifier
# --- ADDED: Import BinaryClassificationEvaluator for AUC ---
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --- ADDED: Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    spark = SparkSession.builder \
        .appName("TrainRandomForest_Weighting") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data_2M.parquet'
    model_output_path = 'models/best_rf_model_weighted'

    logging.info("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # 1. Divisão Treino/Teste
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    logging.info(f"Data split into Training ({train_data.count()} rows) and Test ({test_data.count()} rows).")

    # 2. Calcular o peso
    logging.info("Calculating class weights...")
    num_non_frauds = train_data.filter(col("isFraud") == 0).count()
    num_frauds = train_data.filter(col("isFraud") == 1).count()
    balance_ratio = num_non_frauds / num_frauds if num_frauds > 0 else 1.0 # Avoid division by zero
    logging.info(f"Balance Ratio (non-fraud/fraud): {balance_ratio:.2f}")

    # 3. Adicionar coluna de peso
    train_data_weighted = train_data.withColumn("weight", 
        when(col("isFraud") == 1, balance_ratio).otherwise(1.0)
    )

    # 4. Treinar o modelo
    logging.info("Training Random Forest with 'weightCol'...")
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="isFraud", 
        weightCol="weight",
        numTrees=100, 
        maxDepth=7, 
        seed=42
    )
    
    rf_model = rf.fit(train_data_weighted)

    # 5. Avaliar no conjunto de teste
    logging.info("Starting final evaluation on the unseen test set...")
    predictions = rf_model.transform(test_data)
    predictions.persist() # Optimization

    # --- ENHANCED EVALUATION ---
    
    # a. AUC Calculation
    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    
    # b. Manual Metrics Calculation (including TN for Accuracy)
    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = predictions.filter("prediction = 0.0 AND isFraud = 0").count() # Added TN

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0 # Added Accuracy

    logging.info("\n--- FINAL MODEL REPORT (on Test Data) ---")
    logging.info("Confusion Matrix:")
    predictions.groupBy("isFraud", "prediction").count().show()
    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    
    logging.info("\nPerformance Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC:      {auc:.4f}")
    logging.info(f"Precision (for fraud class): {precision:.4f}")
    logging.info(f"Recall (for fraud class):    {recall:.4f}")
    logging.info(f"F1-Score (for fraud class):  {f1_score:.4f}")
    
    predictions.unpersist()

    rf_model.write().overwrite().save(model_output_path)
    logging.info(f"Best Random Forest model (trained with weighting) saved to '{model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()