# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Benchmark Metrics (Bronze)
# COMMAND ----------

%pip install openpyxl

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd

SOURCE_FILE = "/Volumes/clubos/bronze/raw_uploads/Tema5.benchmark.dataset.xlsx"

def ingest_sheet(sheet_name, target_table):
    """
    Reads an Excel sheet using pandas and converts to Spark DataFrame.
    Works on serverless compute without additional libraries.
    """
    # Read Excel sheet using pandas (pre-installed on Databricks)
    pandas_df = pd.read_excel(SOURCE_FILE, sheet_name=sheet_name)
    
    # Convert pandas DataFrame to Spark DataFrame
    df = spark.createDataFrame(pandas_df)
    
    # Add metadata
    df = df.withColumn("source_file_name", F.lit("Tema5.benchmark.dataset.xlsx"))
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    
    # Write to delta
    df.write.format("delta").mode("overwrite").saveAsTable(f"clubos.bronze.{target_table}")
    print(f"Saved {sheet_name} to clubos.bronze.{target_table} with {df.count()} rows.")

# Ingest the 4 main sheets
ingest_sheet("Main_Website", "bronze_benchmark_main_website")
ingest_sheet("eCommerce", "bronze_benchmark_ecommerce")
ingest_sheet("Streaming", "bronze_benchmark_streaming")
ingest_sheet("Fan_App", "bronze_benchmark_fan_app")
