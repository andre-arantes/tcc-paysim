# src/pipelines/8_evaluate_final_model.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# =================================================================
# ## EDITE ESTAS DUAS LINHAS COM OS MELHORES RESULTADOS DA ETAPA ANTERIOR ##
WINNING_MODEL_PATH = 'models/xgb_best_cv'
WINNING_THRESHOLD = 0.90 # Exemplo: substitua pelo melhor threshold encontrado
# =================================================================

def final_evaluation():
    spark = SparkSession.builder.appName("FinalEvaluation").config("spark.driver.memory", "8g").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("Loading the unseen test data...")
    test_data = spark.read.parquet("data/processed/test_data.parquet")
    
    print(f"Loading the winning XGB model from '{WINNING_MODEL_PATH}'...")
    # Lembre-se de mudar a classe do modelo se o vencedor não for o RF
    model = SparkXGBClassifierModel.load(WINNING_MODEL_PATH)
    
    print("Making final predictions on the test set...")
    test_data_xgb = test_data.withColumnRenamed("isFraud", "label")
    predictions = model.transform(test_data_xgb)

    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    
    # Avaliação com o threshold otimizado
    print("\n" + "="*50)
    print(f"--- FINAL REPORT on Test Set (Model: XGBoost, Threshold: {WINNING_THRESHOLD}) ---")
    print("="*50)
    
    preds_final = predictions.withColumn("prediction_final", 
        when(col("fraud_prob") >= WINNING_THRESHOLD, 1.0).otherwise(0.0)
    )
    
    tp = preds_final.filter("prediction_final = 1.0 AND isFraud = 1").count()
    fp = preds_final.filter("prediction_final = 1.0 AND isFraud = 0").count()
    fn = preds_final.filter("prediction_final = 0.0 AND isFraud = 1").count()
    tn = preds_final.filter("prediction_final = 0.0 AND isFraud = 0").count()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    evaluator_auc = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator_auc.evaluate(predictions)

    print("\nFinal Confusion Matrix:")
    preds_final.withColumnRenamed("label", "isFraud") \
               .groupBy("isFraud", "prediction_final").count().show()
    print(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")

    print("\nFinal Metrics:")
    print(f"AUC:        {auc:.4f}")
    print(f"Precisão:   {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1_score:.4f}")

    spark.stop()

if __name__ == "__main__":
    final_evaluation()