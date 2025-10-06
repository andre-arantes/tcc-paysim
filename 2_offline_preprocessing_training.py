# File: 2_offline_preprocessing_and_validation.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# --- Configuration Flag for Feature Engineering Test ---
# Set to False to test the model's performance without the "golden feature"
USE_SUSPICIOUS_FEATURE = False 

def main():
    """
    Main script for preprocessing, balancing, and robust model validation.
    """

    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset file {subset_path} not found. Run subset creation first.")
    
    spark = SparkSession.builder \
        .appName("OfflineRobustValidation") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()

    subset_path = 'paysim_subset_2M.parquet'
    pipeline_output_path = 'models/preprocessing_pipeline_final'
    model_output_path = 'models/best_rf_model_final'

    print("Loading scaled subset...")
    df = spark.read.parquet(subset_path)
    df_cleaned = df.filter(col("amount") > 0)

    # --- 1. Feature Engineering ---
    print(f"Performing feature engineering... (Using suspicious_amount: {USE_SUSPICIOUS_FEATURE})")
    df_featured = df_cleaned.withColumn(
        "suspicious_amount",
        when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    )

    # --- 2. Data Balancing (Class Weights Method) ---
    fraud_count = df_featured.filter(col("isFraud") == 1).count()
    total_count = df_featured.count()
    non_fraud_count = total_count - fraud_count
    
    balance_ratio = fraud_count / total_count
    
    # Add a weight column to give more importance to the fraud class
    df_featured = df_featured.withColumn("weight", 
        when(col("isFraud") == 1, 1 - balance_ratio).otherwise(balance_ratio)
    )
    print("Added 'weight' column for class balancing.")

    # --- 3. Preprocessing Pipeline ---
    type_indexer = StringIndexer(inputCol="type", outputCol="typeIndex", handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIndex", outputCol="typeVec")
    
    numerical_cols = ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "isFlaggedFraud"]
    if USE_SUSPICIOUS_FEATURE:
        numerical_cols.append("suspicious_amount")

    vec_assembler_num = VectorAssembler(inputCols=numerical_cols, outputCol="numericalFeatures")
    scaler = StandardScaler(inputCol="numericalFeatures", outputCol="scaledNumericalFeatures")
    final_assembler = VectorAssembler(inputCols=["scaledNumericalFeatures", "typeVec"], outputCol="features")

    # Define the full preprocessing pipeline
    preprocessing_pipeline = Pipeline(stages=[type_indexer, type_encoder, vec_assembler_num, scaler, final_assembler])

    print("Fitting the preprocessing pipeline...")
    pipeline_model = preprocessing_pipeline.fit(df_featured)
    
    # Save the fitted pipeline for the streaming app
    pipeline_model.write().overwrite().save(pipeline_output_path)
    print(f"Preprocessing pipeline saved to '{pipeline_output_path}'")
    
    # Transform the data
    df_transformed = pipeline_model.transform(df_featured)

    # --- 4. Model Training with Cross-Validation ---
    rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", weightCol="weight")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [100]) \
        .addGrid(rf.maxDepth, [7]) \
        .build()
        
    evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5, # Using 5 folds for stratified cross-validation
                        seed=42)

    print("Starting 5-fold cross-validation... This may take a while.")
    cv_model = cv.fit(df_transformed)
    
    # --- 5. Report Results ---
    print("\n--- Cross-Validation Results ---")
    avg_f1_scores = cv_model.avgMetrics
    print(f"Average F1-Scores for each parameter grid: {avg_f1_scores}")

    best_model = cv_model.bestModel
    print("\nBest Model Parameters:")
    print(f" - numTrees: {best_model.getNumTrees}")
    print(f" - maxDepth: {best_model.getMaxDepth}")

    # Save the best model
    best_model.write().overwrite().save(model_output_path)
    print(f"\nBest Random Forest model saved to '{model_output_path}'")

    print("\n" + "="*50)
    print("Iniciando treinamento e validação para REGRESSÃO LOGÍSTICA...")
    print("="*50 + "\n")

    lr = LogisticRegression(featuresCol="features", labelCol="isFraud", weightCol="weight")

    # Para a Regressão Logística, os hiperparâmetros são diferentes.
    # Usaremos regParam e elasticNetParam.
    lr_paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1]) \
        .addGrid(lr.elasticNetParam, [0.8]) \
        .build()

    # O avaliador (F1-score) é o mesmo que usamos para o Random Forest
    # Não precisa criar um novo, podemos reutilizar a variável 'evaluator'

    lr_cv = CrossValidator(estimator=lr,
                           estimatorParamMaps=lr_paramGrid,
                           evaluator=evaluator,
                           numFolds=5,
                           seed=42)

    print("Iniciando 5-fold cross-validation para Regressão Logística...")
    lr_cv_model = lr_cv.fit(df_transformed)

    print("\n--- Resultados da Validação Cruzada (Regressão Logística) ---")
    lr_avg_f1_scores = lr_cv_model.avgMetrics
    print(f"F1-Scores médios para cada grid de parâmetros: {lr_avg_f1_scores}")

    best_lr_model = lr_cv_model.bestModel
    print("\nParâmetros do Melhor Modelo (Regressão Logística):")
    print(f" - regParam: {best_lr_model.getRegParam()}")
    print(f" - elasticNetParam: {best_lr_model.getElasticNetParam()}")
    
    # Salvar o melhor modelo de Regressão Logística (opcional, mas boa prática)
    lr_model_output_path = 'models/best_lr_model_final'
    best_lr_model.write().overwrite().save(lr_model_output_path)
    print(f"\nMelhor modelo de Regressão Logística salvo em '{lr_model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()