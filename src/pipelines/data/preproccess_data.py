from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

USE_SUSPICIOUS_FEATURE = True 

def main():
    spark = SparkSession.builder \
        .appName("DataPreprocessing") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    subset_path = 'data/processed/paysim_subset_500k.parquet'
    pipeline_output_path = 'models/preprocessing_pipeline_final'
    transformed_data_path = 'data/processed/transformed_data_500k.parquet'

    print("Loading scaled subset...")
    df = spark.read.parquet(subset_path)
    df_cleaned = df.filter(col("amount") > 0)

    print(f"Performing feature engineering... (Using suspicious_amount: {USE_SUSPICIOUS_FEATURE})")
    df_featured = df_cleaned.withColumn(
        "suspicious_amount",
        when((col("type").isin("TRANSFER", "CASH_OUT")) & (col("amount") > col("oldbalanceOrg")), 1).otherwise(0)
    ).withColumn(
        "errorBalanceOrig",
        col("newbalanceOrig") + col("amount") - col("oldbalanceOrg")
    ) \
    .withColumn(
        "errorBalanceDest",
        col("oldbalanceDest") + col("amount") - col("newbalanceDest")
    )

    type_indexer = StringIndexer(inputCol="type", outputCol="typeIndex", handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIndex", outputCol="typeVec")
    
    numerical_cols = ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "isFlaggedFraud", "errorBalanceOrig", "errorBalanceDest"]
    if USE_SUSPICIOUS_FEATURE:
        numerical_cols.append("suspicious_amount")

    vec_assembler_num = VectorAssembler(inputCols=numerical_cols, outputCol="numericalFeatures")
    scaler = StandardScaler(inputCol="numericalFeatures", outputCol="scaledNumericalFeatures")
    final_assembler = VectorAssembler(inputCols=["scaledNumericalFeatures", "typeVec"], outputCol="features")

    preprocessing_pipeline = Pipeline(stages=[type_indexer, type_encoder, vec_assembler_num, scaler, final_assembler])

    print("Fitting the preprocessing pipeline...")
    pipeline_model = preprocessing_pipeline.fit(df_featured)
    pipeline_model.write().overwrite().save(pipeline_output_path)
    print(f"Preprocessing pipeline saved to '{pipeline_output_path}'")
    
    print("Transforming the entire dataset...")
    df_transformed = pipeline_model.transform(df_featured)

    print(f"Saving transformed data to '{transformed_data_path}'...")
    df_transformed.write.mode("overwrite").parquet(transformed_data_path)
    
    print("\nPreprocessing complete. The transformed data is ready for model training.")
    spark.stop()

if __name__ == "__main__":
    main()
