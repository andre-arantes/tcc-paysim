# File: 5_streaming_consumer_LR.py (Com output melhorado)
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

# A mesma função para processar os alertas
def process_alerts(batch_df, batch_id):
    if batch_df.count() > 0:
        print(f"--- 🚨 ALERTA DE FRAUDE DETECTADA (Batch ID: {batch_id}) 🚨 ---")
        batch_df.show(truncate=False)

def run_streaming_inference_lr():
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection_LR") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    pipeline_path = 'models/preprocessing_pipeline_final'
    model_path = 'models/best_lr_model_final'

    print("Loading saved pipeline and LOGISTIC REGRESSION model...")
    pipeline_model = PipelineModel.load(pipeline_path)
    lr_model = LogisticRegressionModel.load(model_path)

    schema = StructType([
        StructField("step", IntegerType()), StructField("type", StringType()),
        StructField("amount", DoubleType()), StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()), StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()), StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()), StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
    ])

    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .load()

    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    df_transformed = pipeline_model.transform(df_parsed)
    predictions = lr_model.transform(df_transformed)
    fraud_alerts = predictions.filter(col("prediction") == 1.0) \
        .select("step", "type", "amount", "nameOrig", "nameDest", "prediction", "probability")

    # Alteração: Usar foreachBatch para um output limpo
    query = fraud_alerts.writeStream \
        .outputMode("append") \
        .foreachBatch(process_alerts) \
        .start()

    print("--- Listening for transactions on Kafka topic 'transactions' (LOGISTIC REGRESSION) ---")
    print("--- Only fraud alerts will be printed below ---")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_inference_lr()