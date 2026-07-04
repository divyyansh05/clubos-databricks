# Databricks notebook source
# COMMAND ----------
# ClubOS Gold Output - Monthly Brief Inputs
#
# Purpose:
# - collect monthly priority, anomaly, benchmark, and signal context
# - produce deterministic inputs for the Monthly Briefing module

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

priority_board = spark.read.table("clubos.gold.gold_priority_board")
kpi_health = spark.read.table("clubos.gold.gold_kpi_health")
peer_benchmark = spark.read.table("clubos.gold.gold_peer_benchmark")
signals = spark.read.table("clubos.gold.gold_signal_relationships")

months = kpi_health.select("month").distinct()

# 1. Top priorities (top 3 per month)
top_priorities = priority_board.filter(F.col("priority_rank") <= 3).select(
    "month",
    F.struct(
        "priority_rank",
        "priority_id",
        "priority_title",
        "priority_category",
        F.round("priority_score", 4).alias("priority_score")
    ).alias("priority_item")
).groupBy("month").agg(F.collect_list("priority_item").alias("priority_items"))

top_priorities = top_priorities.withColumn(
    "top_priority_ids_json",
    F.to_json(
        F.expr(
            "transform(sort_array(priority_items), x -> named_struct("
            "'priority_id', x.priority_id, "
            "'priority_rank', x.priority_rank, "
            "'priority_title', x.priority_title, "
            "'priority_category', x.priority_category, "
            "'priority_score', x.priority_score"
            "))"
        )
    )
).select("month", "top_priority_ids_json")

# 2. Top anomalies from KPI health (top 3 `review` rows by absolute deviation)
anomaly_window = Window.partitionBy("month").orderBy(F.abs(F.col("deviation_from_rolling_avg")).desc(), F.col("asset_name"), F.col("metric_name"))
top_anomalies = kpi_health.filter(F.col("health_status") == "review").withColumn(
    "anomaly_rank", F.row_number().over(anomaly_window)
).filter(F.col("anomaly_rank") <= 3).select(
    "month",
    F.struct(
        "anomaly_rank",
        "asset_name",
        "metric_name",
        F.round("metric_value", 4).alias("metric_value"),
        F.round("deviation_from_rolling_avg", 4).alias("deviation_from_rolling_avg")
    ).alias("anomaly_item")
).groupBy("month").agg(F.collect_list("anomaly_item").alias("anomaly_items"))

top_anomalies = top_anomalies.withColumn(
    "top_anomalies_json",
    F.to_json(
        F.expr(
            "transform(sort_array(anomaly_items), x -> named_struct("
            "'anomaly_rank', x.anomaly_rank, "
            "'asset_name', x.asset_name, "
            "'metric_name', x.metric_name, "
            "'metric_value', x.metric_value, "
            "'deviation_from_rolling_avg', x.deviation_from_rolling_avg"
            "))"
        )
    )
).select("month", "top_anomalies_json")

# 3. Strongest active signals (assigned to their validated month, empty on other months)
signal_rows = signals.filter(F.col("validation_status") == "active").withColumn(
    "signal_id",
    F.concat_ws(
        "__",
        "source_asset",
        "source_metric",
        "target_asset",
        "target_metric",
        F.col("lag_months").cast("string")
    )
).withColumn(
    "signal_strength_abs", F.abs(F.col("strength_score"))
)

signal_rank_window = Window.partitionBy("last_validated_month").orderBy(F.col("signal_strength_abs").desc(), F.col("signal_id"))
signal_ranked = signal_rows.withColumn("signal_rank", F.row_number().over(signal_rank_window)).filter(F.col("signal_rank") <= 3)

signal_payload = signal_ranked.select(
    F.col("last_validated_month").alias("month"),
    F.struct(
        "signal_rank",
        "signal_id",
        "source_asset",
        "source_metric",
        "target_asset",
        "target_metric",
        "lag_months",
        "relationship_direction",
        F.round("strength_score", 4).alias("strength_score")
    ).alias("signal_item")
).groupBy("month").agg(F.collect_list("signal_item").alias("signal_items"))

signal_payload = signal_payload.withColumn(
    "strongest_signal_ids_json",
    F.to_json(
        F.expr(
            "transform(sort_array(signal_items), x -> named_struct("
            "'signal_rank', x.signal_rank, "
            "'signal_id', x.signal_id, "
            "'source_asset', x.source_asset, "
            "'source_metric', x.source_metric, "
            "'target_asset', x.target_asset, "
            "'target_metric', x.target_metric, "
            "'lag_months', x.lag_months, "
            "'relationship_direction', x.relationship_direction, "
            "'strength_score', x.strength_score"
            "))"
        )
    )
).select("month", "strongest_signal_ids_json")

# 4. Benchmark summary
benchmark_summary = peer_benchmark.groupBy("month").agg(
    F.count("*").alias("benchmarked_metric_count"),
    F.sum(F.when(F.col("rm_rank") >= 4, 1).otherwise(0)).alias("benchmark_underperformance_count"),
    F.round(F.avg("gap_to_peer_median"), 4).alias("avg_gap_to_peer_median"),
    F.round(F.min("gap_to_peer_median"), 4).alias("worst_gap_to_peer_median")
)

benchmark_summary = benchmark_summary.withColumn(
    "benchmark_summary_json",
    F.to_json(
        F.struct(
            "benchmarked_metric_count",
            "benchmark_underperformance_count",
            "avg_gap_to_peer_median",
            "worst_gap_to_peer_median"
        )
    )
).select("month", "benchmark_summary_json")

# 5. Health summary
health_summary = kpi_health.groupBy("month").agg(
    F.count("*").alias("metric_count"),
    F.sum(F.when(F.col("health_status") == "good", 1).otherwise(0)).alias("good_count"),
    F.sum(F.when(F.col("health_status") == "review", 1).otherwise(0)).alias("review_count"),
    F.sum(F.when(F.col("health_status") == "stable", 1).otherwise(0)).alias("stable_count"),
    F.round(F.avg(F.abs(F.col("deviation_from_rolling_avg"))), 4).alias("avg_abs_deviation")
)

health_summary = health_summary.withColumn(
    "health_summary_json",
    F.to_json(
        F.struct(
            "metric_count",
            "good_count",
            "review_count",
            "stable_count",
            "avg_abs_deviation"
        )
    )
).select("month", "health_summary_json")

# 6. Assemble final table with deterministic empty defaults
gold_brief = months.join(top_priorities, on="month", how="left") \
    .join(top_anomalies, on="month", how="left") \
    .join(signal_payload, on="month", how="left") \
    .join(benchmark_summary, on="month", how="left") \
    .join(health_summary, on="month", how="left") \
    .withColumn("top_priority_ids_json", F.coalesce(F.col("top_priority_ids_json"), F.lit("[]"))) \
    .withColumn("top_anomalies_json", F.coalesce(F.col("top_anomalies_json"), F.lit("[]"))) \
    .withColumn("strongest_signal_ids_json", F.coalesce(F.col("strongest_signal_ids_json"), F.lit("[]"))) \
    .withColumn("benchmark_summary_json", F.coalesce(F.col("benchmark_summary_json"), F.lit("{}"))) \
    .withColumn("health_summary_json", F.coalesce(F.col("health_summary_json"), F.lit("{}")))

final_cols = [
    "month",
    "top_priority_ids_json",
    "top_anomalies_json",
    "strongest_signal_ids_json",
    "benchmark_summary_json",
    "health_summary_json",
]

gold_brief = gold_brief.select(*final_cols)
gold_brief.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_monthly_brief_inputs")
print(f"Created gold_monthly_brief_inputs with deterministic monthly briefing payloads. {gold_brief.count()} rows.")
