import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def final_evaluation(model_type, threshold):
    spark_builder = SparkSession.builder \
        .appName(f"FinalEvaluation_{model_type}") \
        .config("spark.driver.memory", "10g") \
        .master("local[*]")

    model_folder_name = ""
    label_col_name = "isFraud" # default label column
    model_class = None

    if model_type.upper() == 'RF':
        model_folder_name = 'rf_best_cv'
        model_class = RandomForestClassificationModel
    elif model_type.upper() == 'LR':
        model_folder_name = 'lr_best_cv'
        model_class = LogisticRegressionModel
    elif model_type.upper() == 'XGB':
        model_folder_name = 'xgb_best_cv'
        model_class = SparkXGBClassifierModel
        label_col_name = "label" # XGBoost expects 'label'
        spark_builder.config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5")
    else:
        logging.error(f"Tipo de modelo '{model_type}' não suportado. Use 'RF', 'LR' ou 'XGB'.")
        return

    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    model_path = f'models/{model_folder_name}'
    test_data_path = "data/processed/test_data.parquet"

    logging.info(f"Loading the unseen test data from '{test_data_path}'...")
    test_data = spark.read.parquet(test_data_path)

    logging.info(f"Loading the winning {model_type.upper()} model from '{model_path}'...")
    try:
        model = model_class.load(model_path)
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo de '{model_path}'. Verifique o caminho e se o modelo foi treinado. Erro: {e}")
        spark.stop()
        return

    logging.info("Making final predictions on the test set...")
    data_to_predict = test_data
    if model_type.upper() == 'XGB': # renomeação apenas para XGBoost
        data_to_predict = test_data.withColumnRenamed("isFraud", "label")

    predictions = model.transform(data_to_predict)
    predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
    predictions.persist()

    logging.info("\n" + "="*50)
    logging.info(f"--- FINAL REPORT on Test Set (Model: {model_type.upper()}, Threshold: {threshold}) ---")
    logging.info("="*50)

    # Apply the chosen threshold
    preds_final = predictions.withColumn("prediction_final",
        when(col("fraud_prob") >= threshold, 1.0).otherwise(0.0)
    )

    tp = preds_final.filter(f"prediction_final = 1.0 AND {label_col_name} = 1").count()
    fp = preds_final.filter(f"prediction_final = 1.0 AND {label_col_name} = 0").count()
    fn = preds_final.filter(f"prediction_final = 0.0 AND {label_col_name} = 1").count()
    tn = preds_final.filter(f"prediction_final = 0.0 AND {label_col_name} = 0").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    evaluator_auc = BinaryClassificationEvaluator(labelCol=label_col_name, metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions) # avalia AUC com probabilidades originais

    logging.info("\nFinal Confusion Matrix:")
    preds_final.withColumnRenamed(label_col_name, "isFraud") \
               .groupBy("isFraud", "prediction_final").count().show()
    logging.info(f"(TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn})")

    logging.info("\nFinal Metrics:")
    logging.info(f"AUC:        {auc:.4f}")
    logging.info(f"Accuracy:   {accuracy:.4f}") 
    logging.info(f"Precision:  {precision:.4f}")
    logging.info(f"Recall:     {recall:.4f}")
    logging.info(f"F1-Score:   {f1_score:.4f}")

    predictions.unpersist()
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform final evaluation on the test set.")
    parser.add_argument('--model', default='XGB', help="Model to evaluate: 'RF', 'LR', or 'XGB'.")
    parser.add_argument('--threshold', type=float, default=0.99, help="Decision threshold to apply.")


    args = parser.parse_args()
    final_evaluation(args.model, args.threshold)