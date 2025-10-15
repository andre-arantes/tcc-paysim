# src/pipelines/6_tune_models.py
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    """
    Script para encontrar os melhores hiperparâmetros para cada modelo
    usando validação cruzada em um conjunto de treino.
    """
    spark = SparkSession.builder \
        .appName("TuneAllModels") \
        .config("spark.driver.memory", "8g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # --- Carregamento e Divisão dos Dados em 3 Partes ---
    transformed_data_path = 'data/processed/transformed_data.parquet'
    print(f"Loading preprocessed data from '{transformed_data_path}'...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # 60% para treino (usado no CrossValidator), 20% para validação (achar threshold), 20% para teste final
    (train_data, validation_data, test_data) = df_transformed.randomSplit([0.6, 0.2, 0.2], seed=42)
    print("Data split into Train (60%), Validation (20%), and Test (20%) sets.")
    
    # Salvar os conjuntos de validação e teste para uso nos próximos scripts
    validation_data.write.mode("overwrite").parquet("data/processed/validation_data.parquet")
    test_data.write.mode("overwrite").parquet("data/processed/test_data.parquet")
    print("Validation and Test sets saved for later use.")

    evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

    # --- 1. Otimização da Regressão Logística ---
    print("\n" + "="*50)
    print("Tuning Logistic Regression...")
    print("="*50)
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")
    lr_paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_paramGrid, evaluator=evaluator, numFolds=3, seed=42)
    lr_cv_model = lr_cv.fit(train_data)
    best_lr_model = lr_cv_model.bestModel
    best_lr_model.write().overwrite().save('models/lr_best_cv')
    print("Best Logistic Regression model saved to 'models/lr_best_cv'")
    print(f"Best params: regParam={best_lr_model.getRegParam()}, elasticNetParam={best_lr_model.getElasticNetParam()}")

    # --- 2. Otimização do Random Forest ---
    print("\n" + "="*50)
    print("Tuning Random Forest...")
    print("="*50)
    rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", seed=42)
    rf_paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 7]).addGrid(rf.numTrees, [100]).build()
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_paramGrid, evaluator=evaluator, numFolds=3, seed=42)
    rf_cv_model = rf_cv.fit(train_data)
    best_rf_model = rf_cv_model.bestModel
    best_rf_model.write().overwrite().save('models/rf_best_cv')
    print("Best Random Forest model saved to 'models/rf_best_cv'")
    print(f"Best params: maxDepth={best_rf_model.getMaxDepth()}, numTrees={best_rf_model.getNumTrees}")
    
    # --- 3. Otimização do XGBoost (COM CORREÇÃO) ---
    print("\n" + "="*50)
    print("Tuning XGBoost...")
    print("="*50)
    
    # Prepara os dados para o XGBoost
    train_data_xgb = train_data.withColumnRenamed("isFraud", "label")

    # --- CORREÇÃO APLICADA AQUI ---
    # Vamos calcular o peso 'scale_pos_weight' como no exemplo do scikit-learn
    num_non_frauds = train_data_xgb.filter(col("label") == 0).count()
    num_frauds = train_data_xgb.filter(col("label") == 1).count()
    
    scale_pos_weight_value = num_non_frauds / num_frauds
    print(f"XGBoost 'scale_pos_weight' calculado: {scale_pos_weight_value:.2f}")

    # Inicializa o classificador com o parâmetro de peso
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        scale_pos_weight=scale_pos_weight_value, # <-- PARÂMETRO CRÍTICO ADICIONADO
        max_depth=7,
        n_estimators=100,
        seed=42
    )
    
    # O treinamento agora usará os pesos
    best_xgb_model = xgb.fit(train_data_xgb)
    best_xgb_model.write().overwrite().save('models/xgb_best_cv')
    print("Best (and correctly weighted) XGBoost model saved to 'models/xgb_best_cv'")

    print("\n✅ All models have been tuned and saved.")
    spark.stop()

if __name__ == "__main__":
    main()