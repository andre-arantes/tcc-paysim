# src/pipelines/7_evaluate_model_in_batch.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_in_batch(model_type, sample_size):
    """
    Executa uma avaliação em batch em uma amostra de dados brutos
    para validar o desempenho do modelo fora do ambiente de streaming.
    """
    spark = SparkSession.builder \
        .appName(f"BatchEvaluation_{model_type}") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- Configuração ---
    raw_data_path = 'data/raw/paysim.csv'
    pipeline_path = 'models/preprocessing_pipeline_final'
    
    if model_type.upper() == 'RF':
        model_path = 'models/best_rf_model_final'
        model_class = RandomForestClassificationModel
    elif model_type.upper() == 'LR':
        model_path = 'models/best_lr_model_final'
        model_class = LogisticRegressionModel
    else:
        print(f"Erro: Tipo de modelo '{model_type}' não suportado. Use 'RF' ou 'LR'.")
        spark.stop()
        return

    # --- Carregamento de Dados e Modelos ---
        print(f"--- Iniciando avaliação em BATCH para o modelo: {model_type.upper()} ---")
    df_raw = spark.read.csv(raw_data_path, header=True, inferSchema=True).limit(sample_size)

    print("Carregando pipeline e modelo treinado...")
    pipeline_model = PipelineModel.load(pipeline_path)
    model = model_class.load(model_path)

    # ... (código de engenharia, pré-processamento e predição é o mesmo) ...
    print("Aplicando engenharia de features...")
    df_featured = df_raw.withColumn("suspicious_amount", when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0))
    print("Aplicando pipeline de pré-processamento...")
    df_transformed = pipeline_model.transform(df_featured)
    print("Realizando predições...")
    predictions = model.transform(df_transformed)
    predictions.persist()
    
    # --- AVALIAÇÃO MANUAL DAS MÉTRICAS (A FORMA CORRETA) ---
    print("\n--- Resultados Finais da Avaliação em Batch (Métricas para Classe Fraude) ---")

    # 1. Matriz de Confusão (como antes)
    print("Matriz de Confusão:")
    predictions.groupBy("isFraud", "prediction").count().show()

    # 2. Cálculo manual dos componentes
    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()
    tn = predictions.filter("prediction = 0.0 AND isFraud = 0").count()

    # 3. Cálculo manual das métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a raw data batch.")
    parser.add_argument('--model', default='RF', help="Model to evaluate: 'RF' (Random Forest) or 'LR' (Logistic Regression).")
    parser.add_argument('--rows', type=int, default=50000, help="Number of rows from the start of the dataset to use for evaluation.")
    
    args = parser.parse_args()
    evaluate_in_batch(args.model, args.rows)