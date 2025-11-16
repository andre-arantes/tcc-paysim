# src/pipelines/training/train_rf_kmeans.py
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TARGET_NON_FRAUD_PERCENTAGE = 0.90 # Mantenha 0.90 para o teste final

def main():
    """
    Treina um modelo Random Forest usando K-means Undersampling
    para balancear o conjunto de treinamento.
    """
    spark = SparkSession.builder \
        .appName(f"TrainRF_KMeans_{int(TARGET_NON_FRAUD_PERCENTAGE*100)}NF") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data_2M.parquet'
    model_output_path = f'models/best_rf_model_kmeans_{int(TARGET_NON_FRAUD_PERCENTAGE*100)}NF' 

    logging.info("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # Use 60/20/20 split: Train / Validation / Test
    (train_data, validation_data, test_data) = df_transformed.randomSplit([0.6, 0.2, 0.2], seed=42)
    logging.info(f"Data split into Training ({train_data.count()} rows), Validation ({validation_data.count()} rows) and Test ({test_data.count()} rows).")
    
    # Persist commonly accessed sets. K-means undersampling will be applied only on `train_data`.
    train_data.persist()
    validation_data.persist()
    test_data.persist()

    fraud_df = train_data.filter(col("isFraud") == 1)
    non_fraud_df = train_data.filter(col("isFraud") == 0)
    
    num_frauds = fraud_df.count()
    num_non_frauds = non_fraud_df.count()
    logging.info(f"Training set composition: {num_frauds} frauds, {num_non_frauds} non-frauds.")

    if num_frauds == 0 or TARGET_NON_FRAUD_PERCENTAGE >= 1.0 or TARGET_NON_FRAUD_PERCENTAGE <= 0.0:
        logging.error("Número de fraudes é zero ou porcentagem alvo inválida. Abortando K-means.")
        k_value = num_non_frauds
    else:
        # Calcula k para atingir a porcentagem desejada de não-fraudes
        target_fraud_percentage = 1.0 - TARGET_NON_FRAUD_PERCENTAGE
        
        # --- CORREÇÃO DE SINTAXE APLICADA ---
        # Derivação matemática (agora como comentários):
        # k / (k + num_frauds) = TARGET_NON_FRAUD_PERCENTAGE 
        # k = TARGET_NON_FRAUD_PERCENTAGE * (k + num_frauds)
        # k * (1 - TARGET_NON_FRAUD_PERCENTAGE) = TARGET_NON_FRAUD_PERCENTAGE * num_frauds
        # k = num_frauds * (TARGET_NON_FRAUD_PERCENTAGE / target_fraud_percentage)
        
        # Implementação final da fórmula:
        k_value = int(num_frauds * (TARGET_NON_FRAUD_PERCENTAGE / target_fraud_percentage))

    k_value = min(k_value, num_non_frauds)
    logging.info(f"Target Non-Fraud Percentage: {TARGET_NON_FRAUD_PERCENTAGE*100:.0f}%")
    logging.info(f"Applying K-means to non-fraud data with k={k_value}...")
    
    kmeans = KMeans(featuresCol="features", k=k_value, seed=42)
    kmeans_model = kmeans.fit(non_fraud_df)
    
    cluster_centers = kmeans_model.clusterCenters()
    
    undersampled_non_fraud_data = [(Vectors.dense(center), 0) for center in cluster_centers]
    undersampled_non_fraud_df = spark.createDataFrame(undersampled_non_fraud_data, ["features", "isFraud"])
    logging.info(f"Created undersampled non-fraud DataFrame with {undersampled_non_fraud_df.count()} rows.")

    balanced_train_df = undersampled_non_fraud_df.union(fraud_df.select("features", "isFraud"))
    logging.info(f"Final balanced training set size: {balanced_train_df.count()} rows.")
    logging.info("Balanced training set class distribution:")
    balanced_train_df.groupBy("isFraud").count().show()
    
    balanced_train_df.persist()
    train_data.unpersist()

    logging.info("Training Random Forest on K-means balanced data...")
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="isFraud", 
        numTrees=100, 
        maxDepth=10, # Mantido maxDepth=10
        seed=42
    )
    
    rf_model = rf.fit(balanced_train_df)
    balanced_train_df.unpersist() 

    logging.info("Starting final evaluation on the original unseen (and unbalanced) test set...")
    predictions = rf_model.transform(test_data)
    predictions.persist()

    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    
    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = predictions.filter("prediction = 0.0 AND isFraud = 0").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    logging.info("\n--- FINAL MODEL REPORT (RF trained with K-means Undersampling) ---")
    logging.info(f"Target Non-Fraud Ratio during K-means: {TARGET_NON_FRAUD_PERCENTAGE*100:.0f}%")
    logging.info("Confusion Matrix:")
    predictions.groupBy("isFraud", "prediction").count().show()
    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    
    logging.info("\nPerformance Metrics (on unbalanced Test Data):")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC:      {auc:.4f}")
    logging.info(f"Precision (fraud): {precision:.4f}")
    logging.info(f"Recall (fraud):    {recall:.4f}")
    logging.info(f"F1-Score (fraud):  {f1_score:.4f}")
    
    predictions.unpersist()
    # cleanup persisted validation and test sets
    validation_data.unpersist()
    test_data.unpersist()

    rf_model.write().overwrite().save(model_output_path)
    logging.info(f"Random Forest model (trained with K-means) saved to '{model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()