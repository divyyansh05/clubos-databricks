# Databricks notebook source
# MAGIC %md
# MAGIC # Normalize Benchmark Metrics (Silver)
# MAGIC 
# MAGIC - Validates month formats
# MAGIC - Standardizes clubs and columns
# MAGIC - Solves the Streaming `digital_active` anomaly (found from auditing)
import pyspark.sql.functions as F
import json

# Define the global path to our metric configuration dictionary
METRIC_DICT_PATH = "/Volumes/clubos/bronze/seeds/metric_dictionary.json"

# In a real environment, you would read this json dynamically
# For databricks execution consistency, we mock the load of allowed names:
ALLOWED_METRICS = {
    "unique_visitors", "visits", "bounce_rate", "recurrence", 
    "conversion_rate", "cart_value", "daily_users", "streamers_rate", 
    "video_play_rate", "app_downloads", "matchday_visits", "heavy_users", "user_rating"
}

def clean_columns(df):
    for col_name in df.columns:
        new_name = col_name.lower().replace(" ", "_").replace(".", "_")
        df = df.withColumnRenamed(col_name, new_name)
    return df

def normalize_benchmark_asset(table_name, expected_asset_name, asset_type):
    df = spark.read.table(f"clubos.bronze.{table_name}")
    df = clean_columns(df)
    
    # Ensure standard month dates
    df = df.withColumn("month", F.to_date(F.col("month")))
    
    # Fix the known bug where streaming had digital_active = 'main_website'
    # We assign the correct canonical product-side label here.
    df = df.withColumn("asset_name", F.lit(expected_asset_name))
    df = df.withColumn("asset_type", F.lit(asset_type))
    
    # Clean up confusing raw fields
    if "digital_active" in df.columns:
        df = df.drop("digital_active")
    if "active_type" in df.columns:
        df = df.drop("active_type")
        
    return df

main_web = normalize_benchmark_asset("bronze_benchmark_main_website", "main_website", "web")
ecomm = normalize_benchmark_asset("bronze_benchmark_ecommerce", "ecommerce", "web")
stream = normalize_benchmark_asset("bronze_benchmark_streaming", "streaming", "streaming")
fan_app = normalize_benchmark_asset("bronze_benchmark_fan_app", "fan_app", "app")

def unpivot_to_fact(df):
    dim_cols = ["month", "club", "asset_name", "asset_type", "source_file_name", "ingestion_timestamp"]
    raw_measure_cols = [c for c in df.columns if c not in dim_cols]
    
    # ENFORCE STRICT ALLOWLIST
    measure_cols = [c for c in raw_measure_cols if c in ALLOWED_METRICS]
    
    if len(measure_cols) == 0:
        return df.sparkSession.createDataFrame([], schema="month date, club string, asset_name string, asset_type string, source_file_name string, ingestion_timestamp timestamp, metric_name string, metric_value double, source_type string")
    
    unpivot_expr = f"stack({len(measure_cols)}, " + ", ".join([f"'{c}', {c}" for c in measure_cols]) + ") AS (metric_name, metric_value)"
    
    fact_df = df.select(*dim_cols, F.expr(unpivot_expr))
    fact_df = fact_df.withColumn("source_type", F.lit("benchmark"))
    return fact_df.filter(F.col("metric_value").isNotNull())

fact_main = unpivot_to_fact(main_web)
fact_ecomm = unpivot_to_fact(ecomm)
fact_stream = unpivot_to_fact(stream)
fact_app = unpivot_to_fact(fan_app)

fact_all = fact_main.unionByName(fact_ecomm).unionByName(fact_stream).unionByName(fact_app)
fact_all.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_benchmark_asset_metrics")
print(f"Created silver_benchmark_asset_metrics with {fact_all.count()} rows.")
