# File: 5_streaming_consumer.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

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

    # Define the schema for the incoming JSON data from Kafka
    schema = StructType([
        StructField("step", IntegerType()),
        StructField("type", StringType()),
        StructField("amount", DoubleType()),
        StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()),
        StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()),
        StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()),
        StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
    ])

    # Read from the Kafka stream
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .load()

    # Parse the JSON value from Kafka
    df_parsed = df_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    # Apply the preprocessing pipeline
    df_transformed = pipeline_model.transform(df_parsed)

    # Make predictions with the Random Forest model
    predictions = rf_model.transform(df_transformed)

    # Filter for fraudulent predictions and select key info
    fraud_alerts = predictions.filter(col("prediction") == 1.0) \
        .select("step", "type", "amount", "nameOrig", "nameDest", "prediction", "probability")

    # Write the alerts to the console
    query = fraud_alerts.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    print("--- Listening for transactions on Kafka topic 'transactions' ---")
    print("--- Fraud alerts will be printed below ---")
    query.awaitTermination()

if __name__ == "__main__":
    run_streaming_inference()