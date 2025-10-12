import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, avg
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (RandomForestClassificationModel, 
                                     LogisticRegressionModel)

def process_and_evaluate_batch(batch_df, batch_id, model_type):
    """
    Processa cada micro-lote para calcular métricas de performance do modelo e do sistema.
    Esta é a implementação da avaliação prequencial em micro-lotes.
    """
    # Ignora lotes vazios
    if batch_df.count() == 0:
        return
        
    print(f"\n\n--- Relatório do Lote (Batch ID: {batch_id}) | Modelo: {model_type.upper()} ---")
    
    # .persist() armazena o batch em memória para evitar recálculos, otimizando a performance
    batch_df.persist()

    # --- 1. Cálculo das Métricas de Performance do Modelo ---
    # Compara a 'prediction' do modelo com o gabarito 'isFraud'
    tp = batch_df.filter("prediction == 1.0 AND isFraud == 1").count()
    fp = batch_df.filter("prediction == 1.0 AND isFraud == 0").count()
    fn = batch_df.filter("prediction == 0.0 AND isFraud == 1").count()
    
    # Cálculo de Precision, Recall e F1-Score para ESTE LOTE
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # --- 2. Cálculo das Métricas Operacionais (Latência) ---
    alerts_df = batch_df.filter("prediction == 1.0")
    alerts_count = alerts_df.count()
    avg_latency_ms = 0.0

    if alerts_count > 0:
        # Calcula a latência para os alertas detectados neste lote
        latencies_df = alerts_df.withColumn("processing_timestamp", lit(time.time())) \
                                .withColumn("latency_sec", col("processing_timestamp") - col("event_timestamp"))
        
        avg_latency_result = latencies_df.agg(avg("latency_sec").alias("avg_latency")).first()
        if avg_latency_result and avg_latency_result["avg_latency"] is not None:
            avg_latency_ms = avg_latency_result["avg_latency"] * 1000

    # --- 3. Exibição do Relatório ---
    print(f"| Transações no Lote: {batch_df.count()}")
    print("| --- Performance Preditiva (Prequencial) ---")
    print(f"| Fraudes Encontradas (TP): {tp} | Alarmes Falsos (FP): {fp} | Fraudes Perdidas (FN): {fn}")
    print(f"| Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    print("| --- Performance Operacional ---")
    print(f"| Alertas de Fraude Gerados: {alerts_count}")
    print(f"| Latência Média dos Alertas: {avg_latency_ms:.2f} ms")
    print("----------------------------------------------------")
    
    # Libera o DataFrame da memória
    batch_df.unpersist()

def run_streaming_evaluation(model_type: str):
    """
    Função principal que configura e executa o pipeline de avaliação em streaming.
    """
    # ... (configuração do Spark Session, modelos, etc. - similar ao script unificado)
    app_name = f"StreamingEvaluation_{model_type.upper()}"
    model_name = f"best_{model_type.lower()}_model_final"
    model_path = f'models/{model_name}'
    checkpoint_path = f'/tmp/spark_checkpoints/evaluation_{model_type.lower()}'
    
    spark = SparkSession.builder.appName(app_name).config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    pipeline_model = PipelineModel.load('models/preprocessing_pipeline_final')
    model = None
    if model_type.lower() == 'rf':
        model = RandomForestClassificationModel.load(model_path)
    elif model_type.lower() == 'lr':
        model = LogisticRegressionModel.load(model_path)
    else:
        raise ValueError("Tipo de modelo inválido. Escolha 'rf' ou 'lr'.")

    # ATUALIZE O SCHEMA para incluir o novo campo de timestamp
    schema = StructType([
        StructField("step", IntegerType()), StructField("type", StringType()),
        StructField("amount", DoubleType()), StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()), StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()), StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()), StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
        StructField("event_timestamp", DoubleType()) # <-- NOVO CAMPO
    ])

    df_stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092").option("subscribe", "transactions").option("startingOffsets", "latest").load()
    
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    df_transformed = pipeline_model.transform(df_parsed)
    predictions = model.transform(df_transformed)

    # O writeStream agora chama nossa função de avaliação detalhada
    query = predictions.writeStream \
        .outputMode("append") \
        .trigger(processingTime='10 seconds') \
        .foreachBatch(lambda df, epoch_id: process_and_evaluate_batch(df, epoch_id, model_type)) \
        .option("checkpointLocation", checkpoint_path) \
        .start()

    print(f"--- Iniciando avaliação em streaming para o modelo {model_type.upper()} ---")
    print("--- Relatórios de performance serão impressos a cada 10 segundos ---")
    query.awaitTermination()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: spark-submit 6_streaming_evaluation.py <model_type>")
        print("  <model_type> pode ser 'rf' (Random Forest) ou 'lr' (Logistic Regression)")
        sys.exit(1)
    
    model_choice = sys.argv[1]
    run_streaming_evaluation(model_choice)