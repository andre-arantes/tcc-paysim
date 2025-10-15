from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    spark = SparkSession.builder.appName("TrainRandomForest_Weighting").config("spark.driver.memory", "8g").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_rf_model_weighted'

    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # 1. Divisão Treino/Teste
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    print(f"Dados divididos em Treinamento ({train_data.count()} linhas) e Teste ({test_data.count()} linhas).")

    # 2. Calcular o peso para a classe minoritária (fraude)
    print("Calculating class weights...")
    num_non_frauds = train_data.filter(col("isFraud") == 0).count()
    num_frauds = train_data.filter(col("isFraud") == 1).count()

    # O peso é a razão entre a classe majoritária e a minoritária
    balance_ratio = num_non_frauds / num_frauds
    print(f"Proporção (não-fraude/fraude): {balance_ratio:.2f}")

    # 3. Adicionar a coluna de peso ao DataFrame de treino
    train_data_weighted = train_data.withColumn("weight", 
        when(col("isFraud") == 1, balance_ratio).otherwise(1.0)
    )

    # 4. Treinar o modelo usando o `weightCol`
    print("Training Random Forest with 'weightCol'...")
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="isFraud", 
        weightCol="weight",
        numTrees=100, 
        maxDepth=7, 
        seed=42
    )
    
    rf_model = rf.fit(train_data_weighted)

    # 5. Avaliar no conjunto de teste (que não tem a coluna de peso)
    print("\n--- Final Evaluation on Unseen (and Unbalanced) Test Data ---")
    predictions = rf_model.transform(test_data)
    
    # Cálculo manual das métricas para a classe de fraude
    tp = predictions.filter("prediction = 1.0 AND isFraud = 1").count()
    fp = predictions.filter("prediction = 1.0 AND isFraud = 0").count()
    fn = predictions.filter("prediction = 0.0 AND isFraud = 1").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Matriz de Confusão no Teste:")
    predictions.groupBy("isFraud", "prediction").count().show()
    print(f"(TP: {tp}, FP: {fp}, FN: {fn})")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    rf_model.write().overwrite().save(model_output_path)
    print(f"\nBest Random Forest model (trained with weighting) saved to '{model_output_path}'")
    
    spark.stop()

if __name__ == "__main__":
    main()