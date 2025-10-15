# src/pipelines/8_evaluate_with_threshold.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_thresholds():
    spark = SparkSession.builder \
        .appName("ThresholdEvaluation") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    # Carrega o modelo treinado sem técnicas de balanceamento para uma análise de threshold justa
    model_path = 'models/best_rf_model_unbalanced'

    print("Loading preprocessed data and the unbalanced model...")
    df_transformed = spark.read.parquet(transformed_data_path)
    try:
        model = RandomForestClassificationModel.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{model_path}'.")
        print("Certifique-se de que você treinou e salvou este modelo primeiro (usando '3_train_rf_unbalanced.py').")
        spark.stop()
        return
    
    # Usaremos o dataset inteiro para ter uma amostra de teste maior
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    
    print("Making predictions to get probabilities...")
    predictions = model.transform(test_data)

    # Extrai a probabilidade da classe positiva (fraude, índice 1)
    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    predictions.persist()

    # Cálculo da AUC Geral
    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
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

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    evaluate_thresholds()