# src/pipelines/7_find_best_threshold.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from xgboost.spark import SparkXGBClassifierModel
from pyspark.ml.functions import vector_to_array

def find_thresholds():
    spark = SparkSession.builder \
        .appName("FindBestThreshold") \
        .config("spark.driver.memory", "10g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("Loading validation data...")
    validation_data = spark.read.parquet("data/processed/validation_data.parquet")
    validation_data.persist()

    models_to_evaluate = {
        "Logistic Regression": LogisticRegressionModel.load('models/lr_best_cv'),
        "Random Forest": RandomForestClassificationModel.load('models/rf_best_cv'),
        "XGBoost": SparkXGBClassifierModel.load('models/xgb_best_cv')
    }
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    for model_name, model in models_to_evaluate.items():
        print("\n" + "="*50)
        print(f"Finding best threshold for: {model_name}")
        print("="*50)

        data_to_predict = validation_data
        label_col_name = "isFraud"
        if model_name == "XGBoost":
            data_to_predict = validation_data.withColumnRenamed("isFraud", "label")
            label_col_name = "label"

        predictions = model.transform(data_to_predict)
        predictions = predictions.withColumn("fraud_prob", vector_to_array(col("probability"))[1])
        
        best_f1 = -1
        best_threshold = -1

        for threshold in thresholds:
            preds_with_threshold = predictions.withColumn("prediction_adj", 
                when(col("fraud_prob") >= threshold, 1.0).otherwise(0.0)
            )
            
            tp = preds_with_threshold.filter(f"prediction_adj = 1.0 AND {label_col_name} = 1").count()
            fp = preds_with_threshold.filter(f"prediction_adj = 1.0 AND {label_col_name} = 0").count()
            fn = preds_with_threshold.filter(f"prediction_adj = 0.0 AND {label_col_name} = 1").count()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Threshold: {threshold:.2f} -> F1-Score: {f1_score:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
        
        print(f"\n🏆 Best Threshold for {model_name}: {best_threshold} (F1-Score: {best_f1:.4f})")

    validation_data.unpersist()
    spark.stop()

if __name__ == "__main__":
    find_thresholds()