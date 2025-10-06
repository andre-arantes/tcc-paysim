# File: 1_data_scaling.py
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
def create_stratified_subset():
    """
    Loads the full PaySim dataset, creates a larger 2M row stratified subset,
    and saves it to Parquet format.
    """
    spark = SparkSession.builder \
        .appName("PaySimScaledSubset") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()

    input_csv_path = 'paysim.csv'
    output_parquet_path = 'paysim_subset_2M.parquet'

    print("Loading the full dataset...")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file {input_csv_path} not found.")
    df = spark.read.csv(input_csv_path, header=True, inferSchema=True)

    total_rows = df.count()
    target_rows = 2_000_000
    
    if target_rows > total_rows:
        target_rows = total_rows

    fraction = target_rows / total_rows
    fractions = {0: fraction, 1: fraction}

    print(f"Performing stratified sampling to get ~{target_rows:,} rows...")
    # Aqui, estamos preservando a proporção original de classes
    subset_df = df.sampleBy("isFraud", fractions, seed=42)

    print("\n--- Original Dataset Distribution ---")
    df.groupBy("isFraud").count().show()

    print("\n--- Scaled Subset Dataset Distribution ---")
    subset_df.groupBy("isFraud").count().show()
    print(f"Total rows in subset: {subset_df.count()}")

    print(f"Saving stratified subset to '{output_parquet_path}'...")
    subset_df.write.mode('overwrite').parquet(output_parquet_path)

    subset_df.coalesce(1).write.csv("paysim_subset_2M.csv", header=True, mode="overwrite")
    print("Subset also saved as CSV.")
    
    print("Successfully created stratified subset.")
    spark.stop()

if __name__ == "__main__":
    create_stratified_subset()