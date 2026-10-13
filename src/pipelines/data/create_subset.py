import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, IntegerType, StringType, 
                             DoubleType)

def create_stratified_subset(input_path, output_path, target_rows):
    spark = SparkSession.builder \
        .appName("PaySimStratifiedSubset") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()

    schema = StructType([
        StructField("step", IntegerType()), StructField("type", StringType()),
        StructField("amount", DoubleType()), StructField("nameOrig", StringType()),
        StructField("oldbalanceOrg", DoubleType()), StructField("newbalanceOrig", DoubleType()),
        StructField("nameDest", StringType()), StructField("oldbalanceDest", DoubleType()),
        StructField("newbalanceDest", DoubleType()), StructField("isFraud", IntegerType()),
        StructField("isFlaggedFraud", IntegerType()),
    ])

    print(f"Loading the full dataset from '{input_path}'...")
    df = spark.read.csv(input_path, header=True, schema=schema)

    total_rows = df.count()
    
    if target_rows > total_rows:
        print(f"Warning: Target rows ({target_rows}) is greater than total rows ({total_rows}). Using all data.")
        target_rows = total_rows

    fraction = target_rows / total_rows
    fractions = {0: fraction, 1: fraction}

    print(f"Performing stratified sampling to get ~{target_rows:,} rows (fraction: {fraction:.4f})...")
    subset_df = df.sampleBy("isFraud", fractions, seed=42)

    print("\n--- Original Dataset Distribution ---")
    df.groupBy("isFraud").count().show()

    print("\n--- Subset Dataset Distribution ---")
    subset_df.groupBy("isFraud").count().show()
    print(f"Total rows in final subset: {subset_df.count()}")

    print(f"Saving stratified subset to '{output_path}'...")
    subset_df.write.mode('overwrite').parquet(output_path)

    output_path_csv = output_path.replace(".parquet", ".csv")
    print(f"Saving stratified subset to '{output_path_csv}' (CSV format)...")
    subset_df.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_path_csv)
     
    print("✅ Successfully created stratified subset.")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified subset of the PaySim dataset.")
    parser.add_argument('--input', default='data/raw/paysim.csv', help='Path to the input CSV file.')
    parser.add_argument('--output', default='data/processed/paysim_subset_1M.parquet', help='Path for the output Parquet file.')
    parser.add_argument('--rows', type=int, default=1_000_000, help='Target number of rows for the subset.')

    args = parser.parse_args()
    create_stratified_subset(args.input, args.output, args.rows)
