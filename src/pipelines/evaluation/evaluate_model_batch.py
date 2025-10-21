# src/pipelines/7_evaluate_model_in_batch.py (Adaptado e Aprimorado)
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from xgboost.spark import SparkXGBClassifierModel
# --- ADICIONADO: Import para AUC ---
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --- ADICIONADO: Configuração do logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_in_batch(model_type, sample_size):
    """
    Executa uma avaliação em batch em uma amostra de dados brutos,
    calculando métricas manualmente para a classe de fraude, além de AUC e Accuracy.
    """
    spark_builder = SparkSession.builder \
        .appName(f"BatchEvaluation_{model_type}") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]")

    # Adiciona o pacote XGBoost apenas se necessário
    if model_type.upper() == 'XGB':
        spark_builder.config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5")

    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- Configuração ---
    raw_data_path = 'data/raw/paysim.csv' # Corrigido o nome do arquivo, se necessário
    # --- CORREÇÃO: Caminho correto para o pipeline salvo ---
    pipeline_path = 'models/preprocessing_pipeline_final'

    # --- ALTERADO: Carregar modelos otimizados pelo CrossValidator ---
    if model_type.upper() == 'RF':
        model_path = 'models/rf_best_cv'
        model_class = RandomForestClassificationModel
    elif model_type.upper() == 'LR':
        model_path = 'models/lr_best_cv'
        model_class = LogisticRegressionModel
    elif model_type.upper() == 'XGB':
        model_path = 'models/xgb_best_cv'
        model_class = SparkXGBClassifierModel
    else:
        logging.error(f"Tipo de modelo '{model_type}' não suportado. Use 'RF', 'LR' ou 'XGB'.")
        spark.stop()
        return

    # --- Carregamento ---
    logging.info(f"--- Iniciando avaliação em BATCH para o modelo: {model_type.upper()} ---")
    logging.info(f"Carregando os primeiros {sample_size:,} registros de '{raw_data_path}'...")
    df_raw = spark.read.csv(raw_data_path, header=True, inferSchema=True).limit(sample_size)

    logging.info(f"Carregando pipeline de pré-processamento de '{pipeline_path}'...")
    try:
        pipeline_model = PipelineModel.load(pipeline_path)
    except Exception as e:
        logging.error(f"Erro ao carregar o pipeline de '{pipeline_path}'. Verifique o caminho. Erro: {e}")
        spark.stop()
        return
        
    logging.info(f"Carregando modelo treinado de '{model_path}'...")
    try:
        model = model_class.load(model_path)
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo de '{model_path}'. Verifique o caminho e se o modelo foi treinado. Erro: {e}")
        spark.stop()
        return

    # --- Pipeline de Inferência ---
    # 1. Engenharia de Features (Exatamente como no consumer)
    logging.info("Aplicando engenharia de features...")
    df_featured = df_raw.withColumn(
        "suspicious_amount", when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig", col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ).withColumn(
        "errorBalanceDest", col("oldbalanceDest") + col("amount") - col("newbalanceDest")
    )

    # 2. Pré-processamento
    logging.info("Aplicando pipeline de pré-processamento...")
    df_transformed = pipeline_model.transform(df_featured)

    # 3. Adaptação para XGBoost e Predição
    label_col_name = "isFraud"
    if model_type.upper() == 'XGB':
        df_transformed = df_transformed.withColumnRenamed("isFraud", "label")
        label_col_name = "label"

    logging.info("Realizando predições (usando threshold padrão 0.5)...")
    predictions = model.transform(df_transformed)
    predictions.persist()

    # --- Avaliação Completa ---
    logging.info("\n--- Resultados Finais (Métricas para Classe Fraude, Threshold 0.5) ---")

    # a. Cálculo da AUC
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label_col_name, metricName="areaUnderROC")
    # Evaluate() espera a coluna 'probability' ou 'rawPrediction'
    auc = evaluator_auc.evaluate(predictions)

    # b. Cálculo manual dos componentes da Matriz de Confusão
    tp = predictions.filter(f"prediction = 1.0 AND {label_col_name} = 1").count()
    fp = predictions.filter(f"prediction = 1.0 AND {label_col_name} = 0").count()
    fn = predictions.filter(f"prediction = 0.0 AND {label_col_name} = 1").count()
    tn = predictions.filter(f"prediction = 0.0 AND {label_col_name} = 0").count()

    # c. Cálculo manual das Métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    logging.info("Matriz de Confusão:")
    predictions.withColumnRenamed(label_col_name, "isFraud") \
               .groupBy("isFraud", "prediction").count().show()

    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    logging.info(f"Acurácia: {accuracy:.4f}")
    logging.info(f"AUC:      {auc:.4f}")
    logging.info(f"Precisão: {precision:.4f}")
    logging.info(f"Recall:   {recall:.4f}")
    logging.info(f"F1-Score: {f1_score:.4f}")

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a raw data batch.")
    parser.add_argument('--model', default='LR', help="Model to evaluate: 'RF', 'LR', or 'XGB'.") # Default para XGB
    parser.add_argument('--rows', type=int, default=50000, help="Number of rows to use.")
    args = parser.parse_args()
    evaluate_in_batch(args.model, args.rows)