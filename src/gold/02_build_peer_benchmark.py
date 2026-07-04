# Databricks notebook source
# MAGIC %md
# MAGIC # Build Peer Benchmark (Gold)
# MAGIC Generates the `gold_peer_benchmark` table.
# MAGIC Calculates medians, ranks, and gaps against peers.
# MAGIC 
# MAGIC **Corrective Update**: The client metric value is explicitly pulled from the internal Real Madrid table.
# MAGIC The peer benchmark table is only used to compute market medians, leaders, and ranks alongside the internal data.

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# Polarity map for benchmark-supported metrics.
# 1 = higher is better, -1 = lower is better.
BENCHMARK_METRIC_POLARITY = {
    "unique_visitors": 1,
    "visits": 1,
    "bounce_rate": -1,
    "recurrence": 1,
    "conversion_rate": 1,
    "cart_value": 1,
    "daily_users": 1,
    "streamers_rate": 1,
    "video_play_rate": 1,
    "matchday_visits": 1,
    "app_downloads": 1,
    "heavy_users": 1,
    "user_rating": 1,
}

polarity_expr = F.create_map([F.lit(x) for kv in BENCHMARK_METRIC_POLARITY.items() for x in kv])

# 1. Get true internal RM values
df_rm = spark.read.table("clubos.silver.silver_internal_asset_metrics") \
    .select("month", "asset_name", "metric_name", F.col("metric_value").alias("rm_value"))

# 2. Get benchmark peer distributions (all benchmark clubs are peers)
df_peers = spark.read.table("clubos.silver.silver_benchmark_asset_metrics") \
    .select("month", "asset_name", "metric_name", F.col("metric_value").alias("peer_value"), "club")

# 2.5 Wire social peer benchmarks into the peer gap scoring pipeline
if spark.catalog.tableExists("clubos.gold.gold_peer_social_benchmark"):
    try:
        social_bench_df = spark.read.table("clubos.gold.gold_peer_social_benchmark")
        social_bench_metrics = ["avg_engagement_per_post", "total_engagement", "instagram_engagement_rate", "posting_frequency_per_day"]
        
        # Create an unpivot/melt expression using stack
        stack_expr = f"stack({len(social_bench_metrics)}, " + ", ".join([f"'{m}', {m}" for m in social_bench_metrics]) + ") as (metric_name, metric_value)"
        
        melted_social = social_bench_df.selectExpr("month", "club_name", stack_expr) \
            .withColumn("asset_name", F.lit("social_media"))
            
        rm_social = melted_social.filter(F.col("club_name") == "real_madrid") \
            .select("month", "asset_name", "metric_name", F.col("metric_value").alias("rm_value"))
            
        peer_social = melted_social.filter(F.col("club_name") != "real_madrid") \
            .select("month", "asset_name", "metric_name", F.col("metric_value").alias("peer_value"), F.col("club_name").alias("club"))
            
        df_rm = df_rm.unionByName(rm_social)
        df_peers = df_peers.unionByName(peer_social)
        print(f"Appended social benchmark data to internal and peer dataframes.")
    except Exception as e:
        print(f"Error processing social benchmarks: {str(e)}")
else:
    print("Skipping social benchmarks: table gold_peer_social_benchmark does not exist yet.")

# 3. Build peer stats from benchmark only (RM is never sourced from benchmark rows)
peer_stats = df_peers.groupBy("month", "asset_name", "metric_name").agg(
    F.expr("percentile_approx(peer_value, 0.5)").alias("peer_median"),
    F.avg("peer_value").alias("peer_mean"),
    F.max("peer_value").alias("peer_max_value"),
    F.min("peer_value").alias("peer_min_value"),
    F.countDistinct("club").alias("club_count")
)

# 4. Keep only metric-month combinations with full peer coverage.
# This also guarantees unsupported internal metrics never appear as benchmarked rows.
# Commercial peers have 5, social peers have 9.
peer_stats = peer_stats.filter(F.col("club_count") >= F.lit(5))

# 5. Align RM internal values to benchmark-supported metrics by month/asset/metric
aligned = df_rm.join(
    peer_stats,
    on=["month", "asset_name", "metric_name"],
    how="inner"
).withColumn("polarity", F.coalesce(polarity_expr.getItem(F.col("metric_name")), F.lit(1)))

# 5b. Choose the correct peer leader by polarity.
aligned = aligned.withColumn(
    "peer_leader_value",
    F.when(F.col("polarity") == -1, F.col("peer_min_value")).otherwise(F.col("peer_max_value"))
).drop("peer_max_value", "peer_min_value")

# 6. Compute RM rank against all peers + RM (1 is best)
rank_base = df_peers.select(
    "month", "asset_name", "metric_name", F.col("peer_value").alias("value")
).join(
    aligned.select("month", "asset_name", "metric_name", "polarity"),
    on=["month", "asset_name", "metric_name"],
    how="inner"
).withColumn("is_rm", F.lit(0))

rank_base = rank_base.unionByName(
    aligned.select("month", "asset_name", "metric_name", F.col("rm_value").alias("value"), "polarity")
    .withColumn("is_rm", F.lit(1))
)

rank_base = rank_base.withColumn("adjusted_value", F.col("value") * F.col("polarity"))
rank_window = Window.partitionBy("month", "asset_name", "metric_name").orderBy(F.col("adjusted_value").desc())
rm_rank_df = rank_base.withColumn("rank_position", F.rank().over(rank_window)) \
    .filter(F.col("is_rm") == 1) \
    .select("month", "asset_name", "metric_name", F.col("rank_position").alias("rm_rank"))

client_df = aligned.join(
    rm_rank_df,
    on=["month", "asset_name", "metric_name"],
    how="left"
)

# 7. Current gap calculations (RM versus peer-only benchmarks)
client_df = client_df.withColumn(
    "gap_to_peer_median",
    F.col("polarity") * (F.col("rm_value") - F.col("peer_median"))
)
client_df = client_df.withColumn(
    "gap_to_leader",
    F.col("polarity") * (F.col("rm_value") - F.col("peer_leader_value"))
).drop("polarity")

# 8. Lookback to calculate 12-month changes for gap and rank
w_time = Window.partitionBy("asset_name", "metric_name").orderBy("month")
client_df = client_df.withColumn("rank_12m_ago", F.lag("rm_rank", 12).over(w_time))
client_df = client_df.withColumn("gap_12m_ago", F.lag("gap_to_peer_median", 12).over(w_time))

# 9. Delta calculations
client_df = client_df.withColumn("rank_change_12m", F.col("rank_12m_ago") - F.col("rm_rank")) # Positive means rank improved
client_df = client_df.withColumn("gap_change_12m", F.col("gap_to_peer_median") - F.col("gap_12m_ago"))

# Select final columns adhering exactly to schema plan
final_cols = [
    "month", "asset_name", "metric_name", "rm_value", 
    "peer_median", "peer_mean", "peer_leader_value", 
    "rm_rank", "club_count", "gap_to_peer_median", 
    "gap_to_leader", "rank_change_12m", "gap_change_12m"
]
gold_df = client_df.select(*final_cols)

# Write to Gold
gold_df.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_peer_benchmark")
print(f"Created gold_peer_benchmark bounded strictly to actual internal RM metrics. {gold_df.count()} rows.")
