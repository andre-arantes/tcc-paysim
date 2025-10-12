# File: 5_streaming_consumer.py (ou consumer_rf.py, com avaliação cumulativa)
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

## --- ADICIONADO: Variáveis globais para acumular as métricas --- ##
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_latency = 0.0
total_messages = 0

## --- ADICIONADO: Função para processar e avaliar cada micro-batch --- ##
def evaluate_and_alert_batch(batch_df, batch_id):
    global total_tp, total_fp, total_tn, total_fn, total_latency, total_messages
    
    # Apenas processa se o batch tiver dados
    if batch_df.count() > 0:
        batch_df.persist()
        
        print(f"\n--- Processando Batch ID: {batch_id} ---")
        
        # 1. Calcula a latência para este batch
        processing_time = time.time()
        batch_latency_df = batch_df.withColumn("latency", processing_time - col("event_timestamp"))
        avg_batch_latency = batch_latency_df.agg({"latency": "avg"}).collect()[0][0]
        
        # Atualiza a latência e contagem total de forma segura
        if avg_batch_latency is not None:
            total_latency += avg_batch_latency * batch_df.count()
            total_messages += batch_df.count()
        
        # 2. Calcula os componentes da matriz de confusão para este batch
        tp = batch_df.filter("prediction = 1.0 AND isFraud = 1").count()
        fp = batch_df.filter("prediction = 1.0 AND isFraud = 0").count()
        tn = batch_df.filter("prediction = 0.0 AND isFraud = 0").count()
        fn = batch_df.filter("prediction = 0.0 AND isFraud = 1").count()
        
        # 3. Atualiza os contadores globais
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        
        # 4. Calcula as métricas CUMULATIVAS (desde o início do stream)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_total_latency = total_latency / total_messages if total_messages > 0 else 0

        print("--- Métricas Cumulativas (Random Forest) ---")
        print(f"Precisão: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        print(f"Latência Média Total: {avg_total_latency:.4f} segundos")
        print(f"(TP: {total_tp}, FP: {total_fp}, FN: {total_fn})")
        
        # 5. Mostra os alertas de fraude (se houver algum neste batch)
        alerts_in_batch = batch_df.filter("prediction = 1.0")
        if alerts_in_batch.count() > 0:
            print(f"--- 🚨 ALERTA(S) DE FRAUDE DETECTADO(S) NO BATCH {batch_id} 🚨 ---")
            alerts_in_batch.select("step", "type", "amount", "nameOrig", "isFraud", "prediction").show(truncate=False)
            
        batch_df.unpersist()

def run_streaming_inference():
    """
    Connects to Kafka, runs real-time inference, and prints fraud alerts.
    """
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    pipeline_path = 'models/preprocessing_pipeline_final'
    model_path = 'models/best_rf_model_final'

    print("Loading saved pipeline and model...")
    pipeline_model = PipelineModel.load(pipeline_path)
    rf_model = RandomForestClassificationModel.load(model_path)

    # ## --- ALTERADO: Schema atualizado para incluir o timestamp do evento --- ##
    schema = StructType([
        StructField("step", IntegerType()), StructField("type", StringType()),
        StructField("amount", DoubleType()), StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()), StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()), StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()), StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
        StructField("event_timestamp", DoubleType()) # Adicionado para cálculo de latência
    ])

    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    print("Applying feature engineering to the stream...")
    df_featured = df_parsed.withColumn(
        "suspicious_amount",
        when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    )

    df_transformed = pipeline_model.transform(df_featured)
    predictions = rf_model.transform(df_transformed)

    # ## --- ALTERADO: Bloco de escrita no console foi substituído pelo foreachBatch --- ##
    # Agora passamos todas as predições para a função de avaliação
    query = predictions.writeStream \
        .outputMode("append") \
        .foreachBatch(evaluate_and_alert_batch) \
        .start()

    print("--- Listening for transactions on Kafka topic 'transactions' ---")
    print("--- Fraud alerts and cumulative metrics will be printed below ---")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_inference()