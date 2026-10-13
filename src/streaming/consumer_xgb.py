# src/streaming/consumer_xgb.py
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql import functions as F  # <--- IMPORT ADICIONADO
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array

# --- Configurações e Variáveis Globais ---

# !! IMPORTANTE: Altere este valor para cada execução do experimento !!
# Ex: 0.99 para o run "Otimizado"
# Ex: 0.50 para o run "Custo Elevado"
BEST_THRESHOLD = 0.99

total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_end_to_end_latency_sum = 0.0 # Soma das latências ponta-a-ponta
total_batch_processing_time_sum = 0.0 # Soma dos tempos de processamento
total_messages = 0

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Função de Avaliação Aprimorada ---
def evaluate_and_alert_batch_xgb(batch_df, batch_id):
    """
    Função foreachBatch corrigida e otimizada.
    1. Mede a latência E2E.
    2. Calcula a matriz de confusão com UMA ÚNICA agregação Spark.
    3. Exibe alertas (I/O).
    4. Mede o tempo total de processamento (incluindo I/O) no FINAL.
    """
    # Marca o início do processamento do batch
    batch_start_time = time.time()
    
    global total_tp, total_fp, total_tn, total_fn, total_end_to_end_latency_sum, total_batch_processing_time_sum, total_messages
    
    batch_size = batch_df.count() # Conta o número de registros no batch
    if batch_size > 0:
        batch_df.persist()
        
        logging.info(f"\n--- Processando Batch ID: {batch_id} (XGBoost @ Thr={BEST_THRESHOLD}) ---")
        logging.info(f"Tamanho do Batch: {batch_size} registros")
        
        # 1. Calcula a LATÊNCIA PONTA-A-PONTA MÉDIA para este batch
        processing_time_for_latency = time.time() # Tempo atual para cálculo da latência
        batch_latency_df = batch_df.withColumn("latency_e2e", processing_time_for_latency - col("event_timestamp"))
        avg_batch_latency_e2e = batch_latency_df.agg({"latency_e2e": "avg"}).collect()[0][0]
        
        if avg_batch_latency_e2e is not None:
            total_end_to_end_latency_sum += avg_batch_latency_e2e * batch_size
            total_messages += batch_size # Incrementa aqui, apenas uma vez por batch
        else:
             avg_batch_latency_e2e = 0.0 # Evita erro NoneType

        logging.info(f"Latência Média E2E (Producer->Consumer) neste Batch: {avg_batch_latency_e2e:.4f} seg")
        
        # 2. Calcula os componentes da matriz de confusão (Otimizado: 1 Job)
        # logging.info("Calculando métricas de classificação...") # (Opcional: verboso)
        metrics = batch_df.select(
            F.sum(F.when(F.col("prediction_final") == 1.0, 1).otherwise(0)).alias("total_pos_preds"),
            F.sum(F.when((F.col("prediction_final") == 1.0) & (F.col("label") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("prediction_final") == 1.0) & (F.col("label") == 0), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("prediction_final") == 0.0) & (F.col("label") == 0), 1).otherwise(0)).alias("tn"),
            F.sum(F.when((F.col("prediction_final") == 0.0) & (F.col("label") == 1), 1).otherwise(0)).alias("fn")
        ).collect()[0]

        tp = metrics["tp"] if metrics["tp"] is not None else 0
        fp = metrics["fp"] if metrics["fp"] is not None else 0
        tn = metrics["tn"] if metrics["tn"] is not None else 0
        fn = metrics["fn"] if metrics["fn"] is not None else 0
        total_pos_preds = metrics["total_pos_preds"] if metrics["total_pos_preds"] is not None else 0
        
        # 3. Atualiza os contadores globais de predição
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn
        
        # 4. Calcula as métricas CUMULATIVAS de PREDIÇÃO
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 5. Mostra os alertas de fraude (usando a contagem já feita)
        # Esta ação (.show()) é o I/O que queremos medir como "Custo de Engenharia"
        if total_pos_preds > 0:
            logging.warning(f"--- 🚨 {total_pos_preds} ALERTA(S) DE FRAUDE DETECTADO(S) NO BATCH {batch_id} (XGBoost @ Thr={BEST_THRESHOLD}) 🚨 ---")
            batch_df.filter("prediction_final = 1.0") \
                    .withColumnRenamed("label", "isFraud") \
                    .select("step", "type", "amount", "nameOrig", "isFraud", "prediction_final") \
                    .show(truncate=False)
        # (Opcional: descomente para logs mais limpos em batches sem fraudes)
        # else:
        #     logging.info("Nenhum alerta de fraude neste batch.")

        # 6. Calcula as métricas de PERFORMANCE (LUGAR CORRETO)
        # O timer é parado AQUI, após todo o processamento, incluindo o .show()
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time
        throughput = batch_size / batch_processing_time if batch_processing_time > 0 else 0
        
        total_batch_processing_time_sum += batch_processing_time

        # 7. Calcula médias CUMULATIVAS de processamento
        avg_total_latency_e2e = total_end_to_end_latency_sum / total_messages if total_messages > 0 else 0
        avg_batch_processing_time = total_batch_processing_time_sum / (batch_id + 1) # Média por batch processado
        avg_throughput = total_messages / total_batch_processing_time_sum if total_batch_processing_time_sum > 0 else 0 # Média geral

        # 8. Loga os resultados do batch e cumulativos
        logging.info(f"Tempo de Processamento deste Batch: {batch_processing_time:.4f} seg")
        logging.info(f"Throughput (Vazão) deste Batch: {throughput:.2f} reg/seg")

        logging.info("--- Métricas Cumulativas (XGBoost) ---")
        logging.info(f"Predição => Precisão: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        logging.info(f"Streaming => Latência E2E Média: {avg_total_latency_e2e:.4f} seg | Proc. Batch Médio: {avg_batch_processing_time:.4f} seg | Throughput Médio: {avg_throughput:.2f} reg/seg")
            
        batch_df.unpersist()

def run_streaming_inference_xgb():
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection_XGB_Final") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    pipeline_path = 'models/preprocessing_pipeline_final'
    model_path = 'models/xgb_best_cv'

    logging.info("Loading saved pipeline and FINAL XGBOOST model...")
    pipeline_model = PipelineModel.load(pipeline_path)
    xgb_model = SparkXGBClassifierModel.load(model_path)

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
    logging.info("Filtering stream (amount > 0) to match training data...")
    df_filtered = df_parsed.filter(col("amount") > 0)

    # 2. (RECOLOCADO) APLICANDO ENGENHARIA DE FEATURES MANUAL
    logging.info("Applying manual feature engineering to the stream...")
    df_featured = df_filtered.withColumn(
        "suspicious_amount",
        when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig",
        col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ) \
    .withColumn(
        "errorBalanceDest",
        col("oldbalanceDest") + col("amount") - col("newbalanceDest")
    )
    df_transformed = pipeline_model.transform(df_featured)
    df_transformed_xgb = df_transformed.withColumnRenamed("isFraud", "label")
    predictions = xgb_model.transform(df_transformed_xgb)

    predictions_with_threshold = predictions \
        .withColumn("fraud_prob", vector_to_array(col("probability"))[1]) \
        .withColumn("prediction_final", when(col("fraud_prob") >= BEST_THRESHOLD, 1.0).otherwise(0.0))

    query = predictions_with_threshold.writeStream \
        .outputMode("append") \
        .foreachBatch(evaluate_and_alert_batch_xgb) \
        .start()

    logging.info(f"--- Iniciando Avaliação em Streaming (XGBOOST Otimizado com Threshold {BEST_THRESHOLD}) ---")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_inference_xgb()