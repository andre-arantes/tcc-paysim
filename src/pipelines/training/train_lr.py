# src/pipelines/4_train_lr.py (Arquivo completo e corrigido para o teste)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    spark = SparkSession.builder \
        .appName("TrainLogisticRegression") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_lr_model_unbalanced'
    
    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    print(f"Dados divididos em Treinamento ({train_data.count()} linhas) e Teste Final ({test_data.count()} linhas).")

    print("\nStarting training for LOGISTIC REGRESSION...")
    
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")

    # --- MUDANÇA CRÍTICA PARA O TESTE ---
    # Vamos treinar um modelo sem regularização para forçá-lo a aprender.
    # Se isso funcionar, sabemos que a regularização era o problema.
    print("Training a model with ZERO regularization to diagnose the issue...")
    lr.setRegParam(0.0)
        
    # Como não estamos mais buscando hiperparâmetros, podemos fazer um .fit() direto
    best_model = lr.fit(train_data)
    
    print("\n--- Model Coefficients (without regularization) ---")
    # Imprime alguns coeficientes para ver se são diferentes de zero
    print(best_model.coefficients[:10])


    # --- Avaliação Final no Conjunto de Teste ---
    print("\n--- Final Evaluation on Unseen Test Data (with default 0.5 threshold) ---")
    predictions = best_model.transform(test_data)
    predictions.persist()

    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Matriz de Confusão no Teste:")
    predictions.groupBy("isFraud", "prediction").count().show()
    print(f"(TP: {tp}, FP: {fp}, FN: {fn})")
    print(f"Precisão Final: {precision:.4f}")
    print(f"Recall Final:   {recall:.4f}")
    print(f"F1-Score Final: {f1_score:.4f}")
    
    predictions.unpersist()

    best_model.write().overwrite().save(model_output_path)
    print(f"\nUNBALANCED Logistic Regression model saved to '{model_output_path}'")

    spark.stop()

if __name__ == "__main__":
    main()