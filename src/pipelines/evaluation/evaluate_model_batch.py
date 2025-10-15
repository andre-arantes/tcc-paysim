# src/pipelines/7_evaluate_model_in_batch.py
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from xgboost.spark import SparkXGBClassifierModel

def evaluate_in_batch(model_type, sample_size):
    """
    Executa uma avaliação em batch calculando as métricas manualmente
    para a classe de fraude.
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
    raw_data_path = 'data/raw/PS_20174392719_1491204439457_log.csv'
    pipeline_path = 'models/preprocessing_pipeline_final'
    
    if model_type.upper() == 'RF':
        model_path = 'models/best_rf_model_weighted' # Usando o modelo treinado com pesos
        model_class = RandomForestClassificationModel
    elif model_type.upper() == 'LR':
        model_path = 'models/best_lr_model_final'
        model_class = LogisticRegressionModel
    elif model_type.upper() == 'XGB':
        model_path = 'models/best_xgb_model_final'
        model_class = SparkXGBClassifierModel
    else:
        print(f"Erro: Tipo de modelo '{model_type}' não suportado. Use 'RF', 'LR' ou 'XGB'.")
        spark.stop()
        return

    # --- Carregamento ---
    print(f"--- Iniciando avaliação em BATCH para o modelo: {model_type.upper()} ---")
    df_raw = spark.read.csv(raw_data_path, header=True, inferSchema=True).limit(sample_size)
    pipeline_model = PipelineModel.load(pipeline_path)
    model = model_class.load(model_path)

    # --- Pipeline de Inferência ---
    df_featured = df_raw.withColumn(
        "suspicious_amount", when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig", col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ).withColumn(
        "errorBalanceDest", col("oldbalanceDest") + col("amount") - col("newbalanceDest")
    )
    
    df_transformed = pipeline_model.transform(df_featured)
    
    label_col_name = "isFraud"
    if model_type.upper() == 'XGB':
        df_transformed = df_transformed.withColumnRenamed("isFraud", "label")
        label_col_name = "label"

    predictions = model.transform(df_transformed)
    predictions.persist()
    
    # --- Avaliação Manual das Métricas ---
    print("\n--- Resultados Finais (Métricas para Classe Fraude) ---")

    tp = predictions.filter(f"prediction = 1.0 AND {label_col_name} = 1").count()
    fp = predictions.filter(f"prediction = 1.0 AND {label_col_name} = 0").count()
    fn = predictions.filter(f"prediction = 0.0 AND {label_col_name} = 1").count()
    tn = predictions.filter(f"prediction = 0.0 AND {label_col_name} = 0").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Matriz de Confusão:")
    # Mostra a matriz com o nome da coluna original para consistência
    predictions.withColumnRenamed(label_col_name, "isFraud") \
               .groupBy("isFraud", "prediction").count().show()
               
    print(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a raw data batch.")
    parser.add_argument('--model', default='RF', help="Model to evaluate: 'RF', 'LR', or 'XGB'.")
    parser.add_argument('--rows', type=int, default=50000, help="Number of rows to use.")
    args = parser.parse_args()
    evaluate_in_batch(args.model, args.rows)