# src/pipelines/9_evaluate_xgb.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_xgb_model():
    """
    Carrega um modelo XGBoost treinado e avalia suas métricas,
    incluindo uma análise de diferentes thresholds de decisão.
    """
    spark = SparkSession.builder \
        .appName("EvaluateXGBoost") \
        .config("spark.driver.memory", "6g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_path = 'models/best_xgb_model_final'

    print(f"Loading XGBoost model from '{model_path}'...")
    model = SparkXGBClassifierModel.load(model_path)
    
    print(f"Loading preprocessed data from '{transformed_data_path}'...")
    df_transformed = spark.read.parquet(transformed_data_path)
    
    # É CRUCIAL usar o mesmo seed para obter o mesmo conjunto de teste
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Making predictions on the test set ({test_data.count()} rows)...")
    # Renomeia o label para o XGBoost poder avaliar corretamente
    test_data_xgb = test_data.withColumnRenamed("isFraud", "label")
    predictions = model.transform(test_data_xgb)

    # Extrai a probabilidade da classe positiva (fraude, índice 1)
    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    predictions.persist()

    # Cálculo da AUC Geral
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    print("\n" + "="*50)
    print(f"Área Sob a Curva ROC (AUC): {auc:.4f}")
    print("="*50)

    # Análise de Thresholds
    thresholds = [0.5, 0.75, 0.90, 0.95, 0.99]

    print("\n--- Evaluating Performance at Different Thresholds ---")
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        
        preds_with_threshold = predictions.withColumn("prediction_adj", 
            when(col("fraud_prob") >= threshold, 1.0).otherwise(0.0)
        )
        
        tp = preds_with_threshold.filter("prediction_adj = 1.0 AND label = 1").count()
        fp = preds_with_threshold.filter("prediction_adj = 1.0 AND label = 0").count()
        fn = preds_with_threshold.filter("prediction_adj = 0.0 AND label = 1").count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"(TP: {tp}, FP: {fp}, FN: {fn})")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    evaluate_xgb_model()