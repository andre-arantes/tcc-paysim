import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# --- MELHORIA DE LOGGING: Configuração do logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    spark = SparkSession.builder \
        .appName("TrainAndTuneLogisticRegression") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_lr_model_tuned' # Novo nome para o modelo otimizado
    
    logging.info("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    logging.info(f"Data split into Training ({train_data.count()} rows) and Final Test ({test_data.count()} rows).")

    logging.info("Starting hyperparameter tuning for LOGISTIC REGRESSION...")
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")

    # --- MELHORIA DE ALGORITMO: Grade de busca robusta ---
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
        
    evaluator_f1_cv = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator_f1_cv, numFolds=3, seed=42)
    
    logging.info("Running 3-fold cross-validation on the training data...")
    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel
    
    logging.info("Hyperparameter tuning complete.")
    logging.info(f"Best regParam found: {best_model.getRegParam()}")
    logging.info(f"Best elasticNetParam found: {best_model.getElasticNetParam()}")

    # --- AVALIAÇÃO FINAL COMPLETA no Conjunto de Teste ---
    logging.info("Starting final evaluation on the unseen test set...")
    predictions = best_model.transform(test_data)
    predictions.persist()

    # --- MELHORIA DE MÉTRICAS: Adicionando AUC ---
    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    
    # Avaliação manual para Precisão e Recall da classe de fraude
    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = predictions.filter("prediction = 0.0 AND isFraud = 0").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

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

    best_model.write().overwrite().save(model_output_path)
    logging.info(f"Best tuned Logistic Regression model saved to '{model_output_path}'")

    spark.stop()

if __name__ == "__main__":
    main()