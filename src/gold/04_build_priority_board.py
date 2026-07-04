# Databricks notebook source
# COMMAND ----------
# ClubOS Gold Output - Priority Board
#
# Purpose:
# - convert scored priority inputs into ranked monthly product outputs
# - persist deterministic evidence payloads for every row

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import json

# Load scoring weights from config
# In production: read from DBFS or a config table
# In development: read from local config file
try:
    with open("/dbfs/FileStore/clubos_config/scoring_config.json") as f:
        SCORING_CONFIG = json.load(f)
except:
    # Fallback to defaults if config not uploaded to DBFS
    SCORING_CONFIG = {
        "formula_weights": {
            "severity": 0.30,
            "persistence": 0.25,
            "peer_gap": 0.20,
            "commercial": 0.15,
            "evidence": 0.10
        }
    }

WEIGHTS = SCORING_CONFIG["formula_weights"]

df = spark.read.table("clubos.gold.gold_priority_inputs")

# 1. Calculate Priority Score
# Formula: weights from scoring_config.json
df = df.withColumn(
    "priority_score",
    (F.col("severity_score") * F.lit(WEIGHTS["severity"]))
    + (F.col("persistence_score") * F.lit(WEIGHTS["persistence"]))
    + (F.col("peer_gap_score") * F.lit(WEIGHTS["peer_gap"]))
    + (F.col("commercial_weight_score") * F.lit(WEIGHTS["commercial"]))
    + (F.col("supporting_evidence_score") * F.lit(WEIGHTS["evidence"]))
)

# 2. Rank priorities per month and keep top 10
w_rank = Window.partitionBy("month").orderBy(F.col("priority_score").desc(), F.col("asset_name"), F.col("metric_name"))
df = df.withColumn("priority_rank", F.row_number().over(w_rank)).filter(F.col("priority_rank") <= 10)

# 3. Deterministic presentation fields (non-AI)
df = df.withColumn("priority_id", F.col("priority_candidate_id"))
df = df.withColumn(
    "priority_title",
    F.concat(F.initcap(F.col("category")), F.lit(" in "), F.initcap(F.regexp_replace(F.col("asset_name"), "_", " ")))
)
df = df.withColumn("primary_metric", F.col("metric_name"))
df = df.withColumn("priority_category", F.col("category"))

df = df.withColumn(
    "summary_text",
    F.concat(
        F.lit(""),
        F.col("primary_metric"),
        F.lit(" is "),
        F.col("trend_direction"),
        F.lit(" versus prior month with trend deviation "),
        F.format_number(F.col("deviation_from_rolling_avg"), 4),
        F.lit(".")
    )
)

df = df.withColumn(
    "why_it_matters",
    F.when(
        F.size("linked_signal_refs") > 0,
        F.lit("This metric is connected to validated leading indicators and can affect commercial outcomes.")
    ).when(
        F.col("peer_context").isNotNull(),
        F.lit("This metric has measurable peer benchmark context and a defined competitive gap.")
    ).otherwise(
        F.lit("This metric is persistently outside stable range and requires operational review.")
    )
)

df = df.withColumn(
    "suggested_next_investigation",
    F.when(F.col("category").like("%conversion%"), F.lit("Investigate funnel drop-off points and recent checkout changes."))
    .when(F.col("category").like("%growth%"), F.lit("Review acquisition channels and campaign pacing for this asset."))
    .when(F.col("category").like("%benchmark%"), F.lit("Compare competitor tactics for this KPI and isolate the largest monthly delta driver."))
    .otherwise(F.lit("Review segment-level drivers and data quality checks before taking action."))
)

# 4. Deterministic evidence payload (fully usable without AI)
evidence_struct = F.struct(
    F.struct(
        F.col("severity_score").alias("severity"),
        F.col("persistence_score").alias("persistence"),
        F.col("peer_gap_score").alias("peer_gap"),
        F.col("commercial_weight_score").alias("commercial_weight"),
        F.col("supporting_evidence_score").alias("supporting_evidence")
    ).alias("score_components"),
    F.struct(
        F.col("metric_value").alias("metric_value"),
        F.col("health_status").alias("health_status"),
        F.col("trend_direction").alias("trend_direction"),
        F.col("deviation_from_rolling_avg").alias("deviation_from_rolling_avg"),
        F.col("seasonal_z_score").alias("seasonal_z_score")
    ).alias("severity_inputs"),
    F.struct(
        F.col("persistence_months").alias("active_months_in_last_3"),
        F.lit(3).alias("lookback_months")
    ).alias("persistence_inputs"),
    F.col("peer_context").alias("peer_context"),
    F.col("linked_signal_refs").alias("linked_signal_references"),
    F.col("supporting_metric_rows").alias("supporting_metric_rows")
)

df = df.withColumn("supporting_metrics_json", F.to_json(evidence_struct))

final_cols = [
    "month",
    "priority_id",
    "priority_title",
    "priority_category",
    "priority_score",
    "priority_rank",
    "asset_name",
    "primary_metric",
    "summary_text",
    "why_it_matters",
    "suggested_next_investigation",
    "supporting_metrics_json"
]

gold_priority = df.select(*final_cols)
gold_priority.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_priority_board")
print(f"Created gold_priority_board with deterministic evidence payloads. {gold_priority.count()} rows.")
