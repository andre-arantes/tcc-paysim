from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    spark = SparkSession.builder \
        .appName("TuneAllModels") \
        .config("spark.driver.memory", "8g") \
        .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.7.5") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    transformed_data_path = 'data/processed/transformed_data_2M.parquet'
    print(f"Loading preprocessed data from '{transformed_data_path}'...")
    df_transformed = spark.read.parquet(transformed_data_path)

    (train_data, validation_data, test_data) = df_transformed.randomSplit([0.6, 0.2, 0.2], seed=42)
    print("Data split into Train (60%), Validation (20%), and Test (20%) sets.")
    
    validation_data.write.mode("overwrite").parquet("data/processed/validation_data.parquet")
    test_data.write.mode("overwrite").parquet("data/processed/test_data.parquet")
    test_data_df = spark.read.parquet("data/processed/test_data.parquet")

    print("Re-particionando para 1 e salvando como CSV único...")
        
    test_data_df.coalesce(1) \
    .write \
    .option("header", "true") \
    .mode("overwrite") \
    .csv("data/processed/test_data_csv_single")

    print("Conjunto de teste salvo. Procure pelo arquivo CSV dentro do diretório 'test_data_csv_single'.")
    
    print("Validation and Test sets saved for later use.")

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="isFraud", metricName="accuracy")
    evaluator_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")

    print("\n" + "="*50)
    print("Tuning Logistic Regression...")
    print("="*50)
    lr = LogisticRegression(featuresCol="features", labelCol="isFraud")
    lr_paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
    lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_paramGrid, evaluator=evaluator_f1, numFolds=3, seed=42)
    lr_cv_model = lr_cv.fit(train_data)
    best_lr_model = lr_cv_model.bestModel
    best_lr_model.write().overwrite().save('models/lr_best_cv')

    
    print("Best Logistic Regression model saved to 'models/lr_best_cv'")
    print(f"Best params: regParam={best_lr_model.getRegParam()}, elasticNetParam={best_lr_model.getElasticNetParam()}")
    print("\n--- Evaluating Best LR Model on Validation Set ---")
    lr_predictions = best_lr_model.transform(validation_data)
    lr_accuracy = evaluator_acc.evaluate(lr_predictions)
    lr_auc = evaluator_auc.evaluate(lr_predictions)
    print(f"Validation Accuracy: {lr_accuracy:.4f}")
    print(f"Validation AUC: {lr_auc:.4f}")

    print("\n" + "="*50)
    print("Tuning Random Forest...")
    print("="*50)
    rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", seed=42)
    rf_paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 7, 9]).addGrid(rf.numTrees, [100]).build()
    rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_paramGrid, evaluator=evaluator_f1, numFolds=3, seed=42)
    rf_cv_model = rf_cv.fit(train_data)
    best_rf_model = rf_cv_model.bestModel
    best_rf_model.write().overwrite().save('models/rf_best_cv')
    print("Best Random Forest model saved to 'models/rf_best_cv'")
    print(f"Best params: maxDepth={best_rf_model.getMaxDepth()}, numTrees={best_rf_model.getNumTrees}")
    print("\n--- Evaluating Best RF Model on Validation Set ---")
    rf_predictions = best_rf_model.transform(validation_data)
    rf_accuracy = evaluator_acc.evaluate(rf_predictions)
    rf_auc = evaluator_auc.evaluate(rf_predictions)
    print(f"Validation Accuracy: {rf_accuracy:.4f}")
    print(f"Validation AUC: {rf_auc:.4f}")
    
    print("\n" + "="*50)
    print("Tuning XGBoost...")
    print("="*50)
    train_data_xgb = train_data.withColumnRenamed("isFraud", "label")
    validation_data_xgb = validation_data.withColumnRenamed("isFraud", "label")
    num_non_frauds = train_data_xgb.filter(col("label") == 0).count()
    num_frauds = train_data_xgb.filter(col("label") == 1).count()
    scale_pos_weight_value = num_non_frauds / num_frauds
    print(f"XGBoost 'scale_pos_weight' calculated: {scale_pos_weight_value:.2f}")
    xgb = SparkXGBClassifier(
        features_col="features", label_col="label",
        scale_pos_weight=scale_pos_weight_value,
        max_depth=7, n_estimators=100, seed=42
    )
    best_xgb_model = xgb.fit(train_data_xgb)
    best_xgb_model.write().overwrite().save('models/xgb_best_cv')
    print("Best (and correctly weighted) XGBoost model saved to 'models/xgb_best_cv'")
    print("\n--- Evaluating Best XGB Model on Validation Set ---")
    evaluator_acc_xgb = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    evaluator_auc_xgb = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    xgb_predictions = best_xgb_model.transform(validation_data_xgb)
    xgb_accuracy = evaluator_acc_xgb.evaluate(xgb_predictions)
    xgb_auc = evaluator_auc_xgb.evaluate(xgb_predictions)
    print(f"Validation Accuracy: {xgb_accuracy:.4f}")
    print(f"Validation AUC: {xgb_auc:.4f}")


    print("\n✅ All models have been tuned and saved.")
    spark.stop()

if __name__ == "__main__":
    main()