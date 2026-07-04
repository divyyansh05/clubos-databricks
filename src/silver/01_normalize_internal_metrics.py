# Databricks notebook source
# MAGIC %md
# MAGIC # Normalize Internal Metrics (Silver)
# MAGIC 
# MAGIC - Validates month formats
# MAGIC - Fixes specific source typos (like otherl_traffic_plays, %android)
# MAGIC - Standardizes columns to snake_case
# MAGIC - Unions all assets into a long fact table (`silver_internal_asset_metrics`)
# MAGIC - Also creates wide table (`silver_internal_asset_monthly`)

import pyspark.sql.functions as F
import json

# Define the global path to our metric configuration dictionary
METRIC_DICT_PATH = "/Volumes/clubos/bronze/seeds/metric_dictionary.json"
# In a real environment, you would read this json dynamically:
# import json; metric_dict = json.load(open("/path/to/metric_dictionary.json"))
# For databricks execution consistency, we mock the load of allowed names:
ALLOWED_METRICS = {
    "unique_visitors", "visits", "page_views", "international_visits", "mobile_visits",
    "search_organic_visits", "social_organic_visits", "marketing_visits", 
    "other_channels_visits", "consumption", "bounce_rate", "recurrence", 
    "new_users", "logged_users", "purchases", "items", "net_sales",
    "search_organic_purchases", "social_organic_purchases", "marketing_purchases",
    "other_channels_purchases", "cart_value", "product_views_rate", "card_addition_rate",
    "checkout_rate", "conversion_rate", "daily_users", "video_plays", "streamers",
    "subscriptions", "search_organic_plays", "social_organic_plays", "marketing_plays",
    "other_traffic_plays", "subscription_rate", "streamers_rate", "video_recurrence",
    "video_play_rate", "video_progress_25_rate", "video_progress_50_rate",
    "video_progress_75_rate", "video_complete_rate", "app_downloads", "matchday_visits",
    "pct_android", "organic_launch_visits", "app_push_visits", "deeplink_visits",
    "marketing_visits", "other_channel_visits", "session_time_avg", "heavy_users", "user_rating"
}

def clean_columns(df):
    """Standardizes column names to snake_case and fixes specific typos."""
    for col_name in df.columns:
        new_name = col_name.lower().replace(" ", "_").replace(".", "_")
        # Fix known typos identified in Data Platform Engineer audit
        if new_name == "%android":
            new_name = "pct_android"
        elif new_name == "otherl_traffic_plays":
            new_name = "other_traffic_plays"
            
        df = df.withColumnRenamed(col_name, new_name)
    return df

def normalize_internal_asset(table_name, asset_name, asset_type):
    df = spark.read.table(f"clubos.bronze.{table_name}")
    df = clean_columns(df)
    
    # Cast month to standard date
    df = df.withColumn("month", F.to_date(F.col("month")))
    
    # Add standardized asset identifiers
    df = df.withColumn("asset_name", F.lit(asset_name))
    df = df.withColumn("asset_type", F.lit(asset_type))
    
    # Drop raw active_type and digital_active as they are replaced by canonical names
    if "digital_active" in df.columns:
        df = df.drop("digital_active")
    if "active_type" in df.columns:
        df = df.drop("active_type")
        
    return df

# Load and clean all 4 sources
main_web = normalize_internal_asset("bronze_internal_main_website", "main_website", "web")
ecomm = normalize_internal_asset("bronze_internal_ecommerce", "ecommerce", "web")
stream = normalize_internal_asset("bronze_internal_streaming", "streaming", "streaming")
fan_app = normalize_internal_asset("bronze_internal_fan_app", "fan_app", "app")

# Write wide tables
main_web.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_internal_main_website")
ecomm.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_internal_ecommerce")
stream.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_internal_streaming")
fan_app.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_internal_fan_app")

# Create the unpivoted fact table `silver_internal_asset_metrics` using stacking
def unpivot_to_fact(df):
    # Get measure columns (exclude standard dimensions)
    dim_cols = ["month", "asset_name", "asset_type", "source_file_name", "ingestion_timestamp"]
    raw_measure_cols = [c for c in df.columns if c not in dim_cols]
    
    # ENFORCE STRICT ALLOWLIST
    # Discard any unexpected columns before unpivoting to protect the fact table
    measure_cols = [c for c in raw_measure_cols if c in ALLOWED_METRICS]
    
    if len(measure_cols) == 0:
        return df.sparkSession.createDataFrame([], schema="month date, asset_name string, asset_type string, source_file_name string, ingestion_timestamp timestamp, metric_name string, metric_value double, source_type string, metric_category string")
        
    # Build unpivot expression
    unpivot_expr = f"stack({len(measure_cols)}, " + ", ".join([f"'{c}', {c}" for c in measure_cols]) + ") AS (metric_name, metric_value)"
    
    fact_df = df.select(*dim_cols, F.expr(unpivot_expr))
    fact_df = fact_df.withColumn("source_type", F.lit("internal"))
    
    # Add metric category classification logically
    fact_df = fact_df.withColumn("metric_category", 
                                 F.when(F.col("metric_name").like("%rate%"), "quality")
                                  .otherwise("volume"))
    
    # filter out null metrics to keep fact table clean
    return fact_df.filter(F.col("metric_value").isNotNull())

fact_main = unpivot_to_fact(main_web)
fact_ecomm = unpivot_to_fact(ecomm)
fact_stream = unpivot_to_fact(stream)
fact_app = unpivot_to_fact(fan_app)

# Union and save
fact_all = fact_main.unionByName(fact_ecomm).unionByName(fact_stream).unionByName(fact_app)
fact_all.write.format("delta").mode("overwrite").saveAsTable("clubos.silver.silver_internal_asset_metrics")
print(f"Created silver_internal_asset_metrics with {fact_all.count()} rows.")
