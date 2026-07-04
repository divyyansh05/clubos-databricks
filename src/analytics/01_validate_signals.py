# Databricks notebook source
# MAGIC %md
# MAGIC # Validate Signals (Analytics)
# MAGIC Generates the `gold_signal_relationships` table by mathematically testing specific business priors across 1, 2, and 3-month lags, and enforcing strict threshold limits.

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# 1. Choose MVP Target Commercial Metrics
TARGETS = [
    {"asset": "ecommerce", "metric": "net_sales"},
    {"asset": "ecommerce", "metric": "conversion_rate"},
    {"asset": "streaming", "metric": "subscriptions"} # Growth oriented
]

# 2. Select Candidate Leading Metrics (Priors supported by business logic)
CANDIDATES = [
    {"asset": "fan_app", "metric": "heavy_users", "interpretation": "Rising app engagement from heavy users predicts increased {} in the following months."},
    {"asset": "main_website", "metric": "bounce_rate", "interpretation": "Increased friction (bounce rate) on the main site tends to degrade {} over a lag window."},
    {"asset": "main_website", "metric": "unique_visitors", "interpretation": "Top-of-funnel traffic volume strongly leads {} shortly after."}
]

df = spark.read.table("clubos.silver.silver_internal_asset_metrics")

# Pivot table temporarily to get columns for correlation logic
# We need month, and then metric values
# df is (month, asset_name, metric_name, metric_value)
# We can create unique keys: asset_metric
df = df.withColumn("feature_key", F.concat_ws("_", "asset_name", "metric_name"))
wide_df = df.groupBy("month").pivot("feature_key").agg(F.max("metric_value"))

# We will calculate lags in PySpark using windowing
w_time = Window.orderBy("month")

results = []

for candidate in CANDIDATES:
    for target in TARGETS:
        source_col = f"{candidate['asset']}_{candidate['metric']}"
        target_col = f"{target['asset']}_{target['metric']}"
        
        # Ensure columns exist in DataFrame (preventing job failures if data is missing)
        if source_col in wide_df.columns and target_col in wide_df.columns:
            for lag in [1, 2, 3]:
                # We need to correlate source(t-lag) with target(t)
                # To do this natively, we lag the target by 'lag' and compare with source
                # Wait: to see if source LEADS target by 1 month:
                # Source at January predicts Target at February.
                # So we pair Source(January) with Target(February). By shifting Target backward 1 month, we align February's target onto January's source.
                shifted_df = wide_df.withColumn(f"{target_col}_shifted_back_{lag}", F.lead(target_col, lag).over(w_time))
                
                # Drop nulls from the comparison pair
                clean_df = shifted_df.select(source_col, f"{target_col}_shifted_back_{lag}").dropna()
                
                # Calculate Pearson Correlation if we have enough data (e.g. > 12 months)
                if clean_df.count() > 12:
                    corr_val = clean_df.corr(source_col, f"{target_col}_shifted_back_{lag}")
                    
                    if corr_val is not None:
                        # 4. Filter for strong stability: We only keep extremely strong commercial relationships > 0.65
                        if abs(corr_val) > 0.65:
                            direction = "positive" if corr_val > 0 else "negative"
                            
                            interpretation = candidate["interpretation"].format(f"{target['asset']} {target['metric']}")
                            
                            results.append({
                                "source_asset": candidate["asset"],
                                "source_metric": candidate["metric"],
                                "target_asset": target["asset"],
                                "target_metric": target["metric"],
                                "lag_months": lag,
                                "relationship_direction": direction,
                                "strength_score": float(corr_val),
                                "validation_status": "active",
                                "business_interpretation": interpretation
                            })

# We need to keep only 2-3 MVP-worthy signals.
# Sort by absolute strength score
sorted_results = sorted(results, key=lambda x: abs(x["strength_score"]), reverse=True)
top_results = sorted_results[:3]

# Create Dataframe and attach dynamic date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
schema = StructType([
    StructField("source_asset", StringType(), True),
    StructField("source_metric", StringType(), True),
    StructField("target_asset", StringType(), True),
    StructField("target_metric", StringType(), True),
    StructField("lag_months", IntegerType(), True),
    StructField("relationship_direction", StringType(), True),
    StructField("strength_score", DoubleType(), True),
    StructField("validation_status", StringType(), True),
    StructField("business_interpretation", StringType(), True)
])

last_validated_month = df.agg(F.max("month").alias("last_validated_month")).first()["last_validated_month"]
final_schema = StructType(schema.fields + [StructField("last_validated_month", DateType(), True)])

if len(top_results) == 0:
    final_df = spark.createDataFrame([], final_schema)
else:
    final_df = spark.createDataFrame(top_results, schema) \
        .withColumn("last_validated_month", F.lit(last_validated_month).cast(DateType()))

# 5. Write to Gold
final_df.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_signal_relationships")
print(f"Registered {final_df.count()} commercial signals to gold_signal_relationships.")
