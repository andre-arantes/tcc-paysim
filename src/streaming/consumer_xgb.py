# Importe a classe de modelo correta
from xgboost.spark import SparkXGBClassifierModel # <-- MUDANÇA AQUI
# ... outras importações ...

def run_streaming_inference_xgb():
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection_XGB") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    # ...
    
    print("Loading saved pipeline and XGBOOST model...")
    pipeline_model = PipelineModel.load(pipeline_path)
    # Carregue o modelo com a classe correta
    xgb_model = SparkXGBClassifierModel.load(model_path) # <-- MUDANÇA AQUI

    # ...
    
    # Renomeie a coluna de label para o XGBoost
    df_transformed = df_transformed.withColumnRenamed("isFraud", "label")

    # Faça predições com o modelo XGBoost
    predictions = xgb_model.transform(df_transformed)
    
    # ... resto do código ...