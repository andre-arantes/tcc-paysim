from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ParquetToCSV_Fix").getOrCreate()

# 1. Colunas que o seu kafka_producer.py espera
# (Eu peguei isso diretamente do seu script do producer)
raw_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

# 2. Ler o conjunto de teste do Parquet
print("Lendo o conjunto de teste do Parquet...")
test_data_df = spark.read.parquet("data/processed/test_data.parquet")

print("DataFrame lido. Schema:")
test_data_df.printSchema() # (Você verá 'typeVec' e 'features' aqui)

# 3. Selecionar APENAS as colunas brutas e salvar como CSV
print("Selecionando colunas brutas e salvando como CSV único...")

# .select(raw_columns) é a correção chave
test_data_df.select(raw_columns) \
    .coalesce(1) \
    .write \
    .option("header", "true") \
    .mode("overwrite") \
    .csv("data/processed/test_data_csv_final")

print("Sucesso. O CSV pronto para o producer está no diretório 'test_data_csv_final'.")