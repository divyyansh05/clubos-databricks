# 01_ingest_internal_metrics.py — placeholder
# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Internal Metrics (Bronze)
# MAGIC Reads the raw internal metrics Excel file, adds lineage metadata, and saves to Bronze.

# COMMAND ----------

%pip install openpyxl

# COMMAND ----------

import pyspark.sql.functions as F
from datetime import datetime
import pandas as pd

# In a real environment, this path would be dynamic or mounted
SOURCE_FILE = "/Volumes/clubos/bronze/raw_uploads/Tema5.internal_metrics.dataset.xlsx"

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
    df = df.withColumn("source_file_name", F.lit("Tema5.internal_metrics.dataset.xlsx"))
    df = df.withColumn("ingestion_timestamp", F.current_timestamp())
    
    # Write to delta
    # spark.sql(f"CREATE DATABASE IF NOT EXISTS clubos_bronze")
    df.write.format("delta").mode("overwrite").saveAsTable(f"clubos.bronze.{target_table}")
    print(f"Saved {sheet_name} to clubos.bronze.{target_table} with {df.count()} rows.")

# COMMAND ----------

# Ingest the 4 main sheets
ingest_sheet("Main_Website", "bronze_internal_main_website")
ingest_sheet("eCommerce", "bronze_internal_ecommerce")
ingest_sheet("Streaming_Website", "bronze_internal_streaming")
ingest_sheet("Fan_App", "bronze_internal_fan_app")
