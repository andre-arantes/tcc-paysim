# File: 4_train_lr.py (Versão Aprimorada)
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    spark = SparkSession.builder.appName("TrainLogisticRegression").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data.parquet'
    model_output_path = 'models/best_lr_model_final'
    
    print("Loading preprocessed data...")
    df_transformed = spark.read.parquet(transformed_data_path)

    # --- MELHORIA 1: Divisão Treino/Teste Final ---
    # Dividimos os dados antes de qualquer treinamento. O 'test_data' ficará intocado.
    (train_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=42)
    print(f"Dados divididos em Treinamento ({train_data.count()} linhas) e Teste Final ({test_data.count()} linhas).")

    print("\nStarting training and hyperparameter tuning for LOGISTIC REGRESSION...")
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud", weightCol="weight")

    # --- MELHORIA 2: Grade de Hiperparâmetros Expandida ---
    # Agora o CrossValidator irá buscar a melhor combinação
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.5, 0.8]) \
        .build()
        
    evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3, seed=42) # Reduzido para 3 folds para ser mais rápido
    
    print("Running 3-fold cross-validation on the training data...")
    # Rodamos o CV apenas nos dados de treino
    cv_model = cv.fit(train_data)

    # O melhor modelo encontrado pela validação cruzada
    best_model = cv_model.bestModel
    
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best regParam: {best_model.getRegParam()}")
    print(f"Best elasticNetParam: {best_model.getElasticNetParam()}")
    print(f"Average F1-Scores on validation folds: {cv_model.avgMetrics}")

    # --- MELHORIA 3: Avaliação Final e Imparcial no Conjunto de Teste ---
    print("\n--- Final Evaluation on Unseen Test Data ---")
    predictions = best_model.transform(test_data)
    final_f1_score = evaluator.evaluate(predictions)
    
    print(f"Final F1-Score on the held-out test set: {final_f1_score:.4f}")

    # Salva o melhor modelo encontrado
    best_model.write().overwrite().save(model_output_path)
    print(f"\nBest Logistic Regression model saved to '{model_output_path}'")

    spark.stop()

if __name__ == "__main__":
    main()