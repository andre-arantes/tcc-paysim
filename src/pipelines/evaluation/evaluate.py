
# src/pipelines/8_evaluate_with_threshold.py (com AUC)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array
# --- ADICIONADO: Importar o avaliador de classificação binária ---
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_thresholds():
    spark = SparkSession.builder.appName("ThresholdEvaluation").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_path = 'models/best_rf_model_weighted'

    print("Loading preprocessed data and the unbalanced model...")
    df_transformed = spark.read.parquet(transformed_data_path)
    model = RandomForestClassificationModel.load(model_path)
    
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    
    print("Making predictions to get probabilities...")
    predictions = model.transform(test_data)

    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    predictions.persist()

    # --- ADICIONADO: Cálculo da AUC ---
    # O avaliador precisa saber qual a coluna de label e a de probabilidade
    # Por padrão, ele usa a coluna "probability" que o modelo gera
    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    
    auc = evaluator_auc.evaluate(predictions)
    print("\n" + "="*50)
    print(f"Área Sob a Curva ROC (AUC): {auc:.4f}")
    print("="*50)
    # --- FIM DA ADIÇÃO ---

    thresholds = [0.5, 0.75, 0.90, 0.95, 0.99]
    print("\n--- Evaluating Performance at Different Thresholds ---")
    # ... (o resto do código permanece o mesmo) ...

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    evaluate_thresholds()