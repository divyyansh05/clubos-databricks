# Databricks notebook source
# MAGIC %md
# MAGIC # Build KPI Health (Gold)
# MAGIC Generates the `gold_kpi_health` table for the Command Center.
# MAGIC Calculates prior periods, rolling averages, and basic health statuses.

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# In a live environment, read from METRIC_DICT_PATH
# For this script's stability, we define the polarity map explicitly
# 1 = higher is better
# -1 = lower is better
# 0 = neutral (e.g. %android)
METRIC_POLARITY = {
    "unique_visitors": 1, "visits": 1, "page_views": 1, "international_visits": 1,
    "mobile_visits": 1, "search_organic_visits": 1, "social_organic_visits": 1,
    "marketing_visits": 1, "other_channels_visits": 1, "consumption": 1,
    "bounce_rate": -1, "recurrence": 1, "new_users": 1, "logged_users": 1,
    "purchases": 1, "items": 1, "net_sales": 1, "search_organic_purchases": 1,
    "social_organic_purchases": 1, "marketing_purchases": 1, "other_channels_purchases": 1,
    "cart_value": 1, "product_views_rate": 1, "card_addition_rate": 1,
    "checkout_rate": 1, "conversion_rate": 1, "daily_users": 1, "video_plays": 1,
    "streamers": 1, "subscriptions": 1, "search_organic_plays": 1, "social_organic_plays": 1,
    "marketing_plays": 1, "other_traffic_plays": 1, "subscription_rate": 1, "streamers_rate": 1,
    "video_recurrence": 1, "video_play_rate": 1, "video_progress_25_rate": 1,
    "video_progress_50_rate": 1, "video_progress_75_rate": 1, "video_complete_rate": 1,
    "app_downloads": 1, "matchday_visits": 1, "pct_android": 0, "organic_launch_visits": 1,
    "app_push_visits": 1, "deeplink_visits": 1, "other_channel_visits": 1,
    "session_time_avg": 1, "heavy_users": 1, "user_rating": 1
}

df = spark.read.table("clubos.silver.silver_internal_asset_metrics")
# Ensure we have month-based windows for time-series calculations
# Partition by asset and metric, order by month
w_time = Window.partitionBy("asset_name", "metric_name").orderBy("month")
w_12m_rolling = Window.partitionBy("asset_name", "metric_name").orderBy("month").rowsBetween(-11, 0)

# Prior month and prior season
df = df.withColumn("prior_month_value", F.lag("metric_value", 1).over(w_time))
df = df.withColumn("prior_season_same_month_value", F.lag("metric_value", 12).over(w_time))

# Rolling 12m avg acts as our recent trend baseline
df = df.withColumn("rolling_12m_avg", F.avg("metric_value").over(w_12m_rolling))

# Deviation from rolling 12-month average (NOT seasonally adjusted)
df = df.withColumn("deviation_from_rolling_avg",
                   F.when(F.col("rolling_12m_avg") != 0,
                          (F.col("metric_value") - F.col("rolling_12m_avg")) / F.col("rolling_12m_avg"))
                    .otherwise(None))

# NEW: Seasonal Z-score - comparing to historical same calendar month
# Extract calendar month (1-12)
df = df.withColumn("calendar_month", F.month(F.col("month")))

# Compute seasonal statistics per metric + calendar month
w_seasonal = Window.partitionBy("asset_name", "metric_name", "calendar_month").orderBy("month").rowsBetween(Window.unboundedPreceding, -1)
df = df.withColumn("seasonal_mean", F.avg("metric_value").over(w_seasonal))
df = df.withColumn("seasonal_std", F.stddev_samp("metric_value").over(w_seasonal))
df = df.withColumn("seasonal_count", F.count("metric_value").over(w_seasonal))

# Compute Z-score only if we have sufficient historical data (at least 2 points)
df = df.withColumn("seasonal_z_score",
                   F.when((F.col("seasonal_count") >= 2) & (F.col("seasonal_std") > 0),
                          (F.col("metric_value") - F.col("seasonal_mean")) / F.col("seasonal_std"))
                    .otherwise(F.lit(0.0)))

# Drop temporary calculation columns
df = df.drop("calendar_month", "seasonal_mean", "seasonal_std", "seasonal_count")

# Trend direction
df = df.withColumn("trend_direction", 
                   F.when(F.col("metric_value") > F.col("prior_month_value"), "up")
                    .when(F.col("metric_value") < F.col("prior_month_value"), "down")
                    .otherwise("flat"))

# Map polarity to dataframe
# We use F.create_map to translate our dictionary dynamically into the spark context
mapping_expr = F.create_map([F.lit(x) for kv in METRIC_POLARITY.items() for x in kv])
df = df.withColumn("polarity", mapping_expr.getItem(F.col("metric_name")).cast(IntegerType()))
df = df.fillna({"polarity": 1}) # default to higher-is-better if missing

# Health status
# Metric-aware status logic utilizing polarity
df = df.withColumn("health_status",
                   F.when(F.col("polarity") == 1,
                          F.when(F.col("deviation_from_rolling_avg") > 0.05, "good")
                           .when(F.col("deviation_from_rolling_avg") < -0.05, "review")
                           .otherwise("stable"))
                    .when(F.col("polarity") == -1,
                          F.when(F.col("deviation_from_rolling_avg") < -0.05, "good")
                           .when(F.col("deviation_from_rolling_avg") > 0.05, "review")
                           .otherwise("stable"))
                    .otherwise("stable") # neutral polarity
                  )

# Select final columns adhering exactly to schema plan
final_cols = [
    "month", "asset_name", "metric_name", "metric_value",
    "prior_month_value", "prior_season_same_month_value", "rolling_12m_avg",
    "deviation_from_rolling_avg", "seasonal_z_score",
    "trend_direction", "health_status"
]
gold_df = df.select(*final_cols)

# Write to Gold
gold_df.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_kpi_health")
print(f"Created gold_kpi_health with {gold_df.count()} rows.")
