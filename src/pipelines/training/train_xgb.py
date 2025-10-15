# src/pipelines/5_train_xgb.py
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier

def main():
    """
    Script dedicado apenas ao treinamento do modelo XGBoost.
    """
    spark = SparkSession.builder \
        .appName("TrainXGBoost") \
        .config("spark.driver.memory", "6g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_xgb_model_final'

    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # 1. Divisão Treino/Teste
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    print(f"Dados divididos em Treinamento ({train_data.count()} linhas) e Teste ({test_data.count()} linhas).")

    # 2. Renomear colunas para o padrão do XGBoost
    train_data_xgb = train_data.withColumnRenamed("isFraud", "label")

    # 3. Inicializar e Treinar o Modelo
    print("Initializing XGBoost Classifier...")
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        max_depth=7,
        n_estimators=100,
        seed=42
    )
    
    print("Training XGBoost model on the training data... This might take a while.")
    xgb_model = xgb.fit(train_data_xgb)

    # 4. Salvar o modelo treinado
    xgb_model.write().overwrite().save(model_output_path)
    print(f"\n✅ XGBoost model successfully trained and saved to '{model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()