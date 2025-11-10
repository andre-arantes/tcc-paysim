# src/streaming/consumer_rf.py (Com métricas de streaming detalhadas)
import time
import logging # Usar logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType,
                             DoubleType)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
# --- ADICIONADO: Para extrair probabilidade ---
from pyspark.ml.functions import vector_to_array

# --- Configuração do Melhor Modelo ---
# Substitua 0.5 pelo MELHOR threshold encontrado para RF no script 7_find_best_threshold.py
# (No seu log anterior, o melhor F1 para RF foi com threshold 0.5)
BEST_THRESHOLD = 0.5

# --- Variáveis globais para acumular as métricas ---
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_end_to_end_latency_sum = 0.0 # Renomeado para clareza
total_batch_processing_time_sum = 0.0 # Adicionado
total_messages = 0

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Função de Avaliação Aprimorada ---
def evaluate_and_alert_batch_rf(batch_df, batch_id): # Renomeado para clareza
    # Marca o início do processamento do batch
    batch_start_time = time.time()

    # Declara que vamos modificar as variáveis globais
    global total_tp, total_fp, total_tn, total_fn, total_end_to_end_latency_sum, total_batch_processing_time_sum, total_messages

    batch_size = batch_df.count()
    if batch_size > 0:
        batch_df.persist()

        logging.info(f"\n--- Processando Batch ID: {batch_id} (Random Forest @ Thr={BEST_THRESHOLD}) ---")
        logging.info(f"Tamanho do Batch: {batch_size} registros")

        # 1. Calcula a LATÊNCIA PONTA-A-PONTA MÉDIA para este batch
        processing_time_for_latency = time.time()
        batch_latency_df = batch_df.withColumn("latency_e2e", processing_time_for_latency - col("event_timestamp"))
        # Tratamento de erro caso a agregação retorne None (batch vazio após filtros internos?)
        agg_result = batch_latency_df.agg({"latency_e2e": "avg"}).collect()
        avg_batch_latency_e2e = agg_result[0][0] if agg_result and agg_result[0][0] is not None else 0.0

        if avg_batch_latency_e2e is not None:
             total_end_to_end_latency_sum += avg_batch_latency_e2e * batch_size
             total_messages += batch_size # Incrementa aqui, apenas uma vez por batch
        else:
             avg_batch_latency_e2e = 0.0

        logging.info(f"Latência Média E2E (Producer->Consumer) neste Batch: {avg_batch_latency_e2e:.4f} seg")

        # 2. Calcula os componentes da matriz de confusão (usando prediction_final)
        tp = batch_df.filter("prediction_final = 1.0 AND isFraud = 1").count()
        fp = batch_df.filter("prediction_final = 1.0 AND isFraud = 0").count()
        tn = batch_df.filter("prediction_final = 0.0 AND isFraud = 0").count()
        fn = batch_df.filter("prediction_final = 0.0 AND isFraud = 1").count()

        # 3. Atualiza os contadores globais de predição
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

        # 4. Calcula as métricas CUMULATIVAS de PREDIÇÃO
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 5. Calcula as métricas CUMULATIVAS de PERFORMANCE do STREAMING
        avg_total_latency_e2e = total_end_to_end_latency_sum / total_messages if total_messages > 0 else 0

        # Calcula o TEMPO DE PROCESSAMENTO e THROUGHPUT deste batch
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time
        throughput = batch_size / batch_processing_time if batch_processing_time > 0 else 0
        total_batch_processing_time_sum += batch_processing_time

        avg_batch_processing_time = total_batch_processing_time_sum / (batch_id + 1)
        avg_throughput = total_messages / total_batch_processing_time_sum if total_batch_processing_time_sum > 0 else 0

        logging.info(f"Tempo de Processamento deste Batch: {batch_processing_time:.4f} seg")
        logging.info(f"Throughput (Vazão) deste Batch: {throughput:.2f} reg/seg")

        logging.info("--- Métricas Cumulativas (Random Forest) ---")
        logging.info(f"Predição => Precisão: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        logging.info(f"Streaming => Latência E2E Média: {avg_total_latency_e2e:.4f} seg | Proc. Batch Médio: {avg_batch_processing_time:.4f} seg | Throughput Médio: {avg_throughput:.2f} reg/seg")

        # 6. Mostra os alertas de fraude (usando prediction_final)
        alerts_in_batch = batch_df.filter("prediction_final = 1.0")
        if alerts_in_batch.count() > 0:
            logging.warning(f"--- 🚨 ALERTA(S) DE FRAUDE DETECTADO(S) NO BATCH {batch_id} (RF @ Thr={BEST_THRESHOLD}) 🚨 ---")
            # Seleciona as colunas relevantes, incluindo a predição final
            alerts_in_batch.select("step", "type", "amount", "nameOrig", "isFraud", "prediction_final").show(truncate=False)

        batch_df.unpersist()


def run_streaming_inference_rf(): # Renomeado para clareza
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection_RF_Final") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1") \
        .config("spark.driver.memory", "8g") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR") # Reduz verbosidade

    pipeline_path = 'models/preprocessing_pipeline_final'
    model_path = 'models/rf_best_cv' # Carrega o modelo otimizado

    logging.info("Loading saved pipeline and FINAL RANDOM FOREST model...")
    pipeline_model = PipelineModel.load(pipeline_path)
    rf_model = RandomForestClassificationModel.load(model_path)

    # Schema completo incluindo timestamp
    schema = StructType([
        StructField("step", IntegerType()), StructField("type", StringType()),
        StructField("amount", DoubleType()), StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()), StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()), StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()), StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
        StructField("event_timestamp", DoubleType())
    ])

    df_stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092").option("subscribe", "transactions").option("startingOffsets", "latest").load()
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    logging.info("Applying complete feature engineering to the stream...")
    df_featured = df_parsed.withColumn(
        "suspicious_amount", when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig", col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ).withColumn(
        "errorBalanceDest", col("oldbalanceDest") + col("amount") - col("newbalanceDest")
    )

    df_transformed = pipeline_model.transform(df_featured)
    predictions = rf_model.transform(df_transformed)

    # --- Aplica o threshold otimizado ---
    predictions_with_threshold = predictions \
        .withColumn("fraud_prob", vector_to_array(col("probability"))[1]) \
        .withColumn("prediction_final", when(col("fraud_prob") >= BEST_THRESHOLD, 1.0).otherwise(0.0))

    # Chama a função de avaliação aprimorada
    query = predictions_with_threshold.writeStream \
        .outputMode("append") \
        .foreachBatch(evaluate_and_alert_batch_rf) \
        .start()

    logging.info(f"--- Iniciando Avaliação em Streaming (RANDOM FOREST Otimizado com Threshold {BEST_THRESHOLD}) ---")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_inference_rf() # Chama a função renomeada