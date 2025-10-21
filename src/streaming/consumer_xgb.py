# src/streaming/consumer_xgb.py
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array

BEST_THRESHOLD = 0.95

total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_latency = 0.0
total_messages = 0

# --- ADICIONADO: Função de avaliação completa ---
def evaluate_and_alert_batch_xgb(batch_df, batch_id):
    # Declara que vamos modificar as variáveis globais
    global total_tp, total_fp, total_tn, total_fn, total_latency, total_messages
    
    if batch_df.count() > 0:
        batch_df.persist()
        
        print(f"\n--- Processando Batch ID: {batch_id} (XGBoost com Threshold Otimizado) ---")
        
        # 1. Calcula a latência para este batch
        processing_time = time.time()
        batch_latency_df = batch_df.withColumn("latency", processing_time - col("event_timestamp"))
        avg_batch_latency = batch_latency_df.agg({"latency": "avg"}).collect()[0][0]
        
        if avg_batch_latency is not None:
            total_latency += avg_batch_latency * batch_df.count()
            total_messages += batch_df.count()
        
        # 2. Calcula os componentes da matriz de confusão
        tp = batch_df.filter("prediction_final = 1.0 AND label = 1").count()
        fp = batch_df.filter("prediction_final = 1.0 AND label = 0").count()
        tn = batch_df.filter("prediction_final = 0.0 AND label = 0").count()
        fn = batch_df.filter("prediction_final = 0.0 AND label = 1").count()
        
        # 3. Atualiza os contadores globais
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn
        
        # 4. Calcula as métricas CUMULATIVAS
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_total_latency = total_latency / total_messages if total_messages > 0 else 0
        
        print("--- Métricas Cumulativas (XGBoost) ---")
        print(f"Precisão: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        print(f"Latência Média Total: {avg_total_latency:.4f} segundos")
        
        # 5. Mostra os alertas de fraude
        alerts_in_batch = batch_df.filter("prediction = 1.0")
        if alerts_in_batch.count() > 0:
            print(f"--- 🚨 ALERTA(S) DE FRAUDE DETECTADO(S) NO BATCH {batch_id} 🚨 ---")
            alerts_in_batch.withColumnRenamed("label", "isFraud") \
                           .select("step", "type", "amount", "nameOrig", "isFraud", "prediction").show(truncate=False)
            
        batch_df.unpersist()

def run_streaming_inference_xgb():
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection_XGB") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    pipeline_path = 'models/preprocessing_pipeline_final'
    model_path = 'models/xgb_best_cv'

    print("Loading saved pipeline and XGBOOST model...")
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

    print("Applying complete feature engineering to the stream...")
    df_featured = df_parsed.withColumn(
        "suspicious_amount",
        when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig",
        col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ).withColumn(
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

    print(f"--- Iniciando Avaliação em Streaming (XGBOOST Otimizado com Threshold {BEST_THRESHOLD}) ---")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_inference_xgb()