# Databricks notebook source
# COMMAND ----------
# ClubOS Analytics - Compute Priority Inputs
#
# Purpose:
# - calculate deterministic scoring inputs for the Priority Board
# - preserve a inspectable evidence chain for every candidate row

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import json

# 1. Load Gold tables
kpi_health = spark.read.table("clubos.gold.gold_kpi_health")
peer_benchmark = spark.read.table("clubos.gold.gold_peer_benchmark")
signals = spark.read.table("clubos.gold.gold_signal_relationships")

# 2. Restrict signal references to mathematically validated signals only
active_signals = signals.filter(F.col("validation_status") == "active")

source_signal_refs = active_signals.withColumn(
    "metric_key", F.concat_ws("_", "source_asset", "source_metric")
).groupBy("metric_key").agg(
    F.collect_list(
        F.struct(
            F.lit("source").alias("signal_role"),
            "source_asset",
            "source_metric",
            "target_asset",
            "target_metric",
            "lag_months",
            "relationship_direction",
            "strength_score",
            "validation_status",
            "business_interpretation"
        )
    ).alias("source_signal_refs")
)

target_signal_refs = active_signals.withColumn(
    "metric_key", F.concat_ws("_", "target_asset", "target_metric")
).groupBy("metric_key").agg(
    F.collect_list(
        F.struct(
            F.lit("target").alias("signal_role"),
            "source_asset",
            "source_metric",
            "target_asset",
            "target_metric",
            "lag_months",
            "relationship_direction",
            "strength_score",
            "validation_status",
            "business_interpretation"
        )
    ).alias("target_signal_refs")
)

# 3. Join KPI health with benchmark context at the monthly metric grain
df = kpi_health.alias("h").join(
    peer_benchmark.alias("b"),
    on=["month", "asset_name", "metric_name"],
    how="left"
).select(
    "month",
    "asset_name",
    "metric_name",
    "metric_value",
    "health_status",
    "trend_direction",
    "deviation_from_rolling_avg",
    "seasonal_z_score",
    F.col("b.rm_rank").alias("peer_rank"),
    F.col("b.club_count").alias("peer_club_count"),
    F.col("b.peer_median"),
    F.col("b.peer_leader_value"),
    F.col("b.gap_to_peer_median"),
    F.col("b.gap_to_leader")
)

# 4. Score inputs
df = df.withColumn("metric_key", F.concat_ws("_", "asset_name", "metric_name"))
df = df.withColumn("is_active", F.when(F.col("health_status") != "stable", F.lit(1)).otherwise(F.lit(0)))

# NEW SEVERITY CALCULATION: Use seasonal Z-score instead of rolling avg deviation
# Formula: severity = min(1.0, abs(z_score) / 2.0)
# Z-score of 0 = no deviation = severity 0.0
# Z-score of 1.0 = 1 std dev = severity 0.5
# Z-score of 2.0+ = 2+ std devs = severity 1.0 (maximum)
df = df.withColumn(
    "severity_score",
    F.when(
        F.col("health_status") != "stable",
        F.least(F.lit(1.0), F.abs(F.col("seasonal_z_score")) / F.lit(2.0))
    ).otherwise(F.lit(0.0))
)

w_persistence = Window.partitionBy("asset_name", "metric_name").orderBy("month").rowsBetween(-2, 0)
df = df.withColumn("persistence_months", F.sum("is_active").over(w_persistence))
df = df.withColumn("persistence_score", F.least(F.lit(1.0), F.col("persistence_months") / F.lit(3.0)))

df = df.withColumn(
    "peer_gap_score",
    F.when(F.col("peer_rank").isNull(), F.lit(0.0))
    .when(F.col("peer_rank") >= 5, F.lit(1.0))
    .when(F.col("peer_rank") == 4, F.lit(0.8))
    .when(F.col("peer_rank") == 3, F.lit(0.4))
    .otherwise(F.lit(0.0))
)

# 5. Commercial weight from metric_dictionary.json
# Load metric dictionary for commercial weights
try:
    with open("/dbfs/FileStore/clubos_config/metric_dictionary.json") as f:
        metric_dict = json.load(f)
except:
    # Fallback to empty dict if not in DBFS
    metric_dict = {}

# Build commercial weight DataFrame
# Note: metric_dictionary keys are metric names only (not asset_metric)
# We need to extract just the metric name from metric_key for lookup
commercial_weights_list = [
    (metric_name, float(props.get("commercial_weight", 0.4)))
    for metric_name, props in metric_dict.items()
    if "commercial_weight" in props
]

if commercial_weights_list:
    weights_df = spark.createDataFrame(commercial_weights_list, ["metric_name", "commercial_weight_score"])
    df = df.join(weights_df, on="metric_name", how="left")
    # Fill missing with default 0.4
    df = df.fillna(0.4, subset=["commercial_weight_score"])
else:
    # Fallback if metric_dictionary not loaded
    validated_source_keys = source_signal_refs.select("metric_key").withColumn("is_validated_source", F.lit(1))
    df = df.join(validated_source_keys, on="metric_key", how="left")
    df = df.withColumn(
        "commercial_weight_score",
        F.when(F.col("is_validated_source") == 1, F.lit(0.8)).otherwise(F.lit(0.4))
    )

# 6. Keep only actionable rows (non-stable health) and assign categories
df = df.filter(F.col("is_active") == 1)

df = df.withColumn(
    "category",
    F.when((F.col("health_status") == "review") & (F.col("asset_name") == "ecommerce"), F.lit("conversion weakness"))
    .when(
        (F.col("health_status") == "review")
        & (F.col("metric_name").like("%visitors%") | F.col("metric_name").like("%app_downloads%")),
        F.lit("growth risk")
    )
    .when(F.col("peer_gap_score") >= 0.8, F.lit("benchmark underperformance"))
    .when(
        (F.col("health_status") == "good")
        & (F.col("metric_name").like("%recurrence%") | F.col("metric_name").like("%heavy_users%")),
        F.lit("engagement opportunity")
    )
    .when(F.col("health_status") == "review", F.lit("resilience concern"))
    .otherwise(F.lit("engagement opportunity"))
)

# 7. Build supporting metric evidence rows at month+asset grain
supporting_pool = df.select(
    "month",
    "asset_name",
    F.struct(
        "metric_name",
        "metric_value",
        "health_status",
        "trend_direction",
        "deviation_from_rolling_avg",
        "seasonal_z_score",
        "severity_score"
    ).alias("supporting_row")
).groupBy("month", "asset_name").agg(F.collect_list("supporting_row").alias("asset_supporting_rows"))

df = df.join(supporting_pool, on=["month", "asset_name"], how="left")
df = df.withColumn(
    "supporting_metric_rows",
    F.expr("filter(asset_supporting_rows, x -> x.metric_name <> metric_name)")
).drop("asset_supporting_rows")

# NEW EVIDENCE SCORING: Scale with count instead of binary
# 0 metrics = 0.0, 1 metric = 0.2, 5+ metrics = 1.0
EVIDENCE_MAX_COUNT = 5
df = df.withColumn(
    "supporting_evidence_score",
    F.least(F.lit(1.0), F.size("supporting_metric_rows") / F.lit(EVIDENCE_MAX_COUNT))
)

# 8. Attach linked signal references for source and target perspectives
SIGNAL_REF_SCHEMA = """
array<struct<
    signal_role:string,
    source_asset:string,
    source_metric:string,
    target_asset:string,
    target_metric:string,
    lag_months:int,
    relationship_direction:string,
    strength_score:double,
    validation_status:string,
    business_interpretation:string
>>
"""
empty_signal_refs = F.from_json(F.lit("[]"), SIGNAL_REF_SCHEMA)

df = df.join(source_signal_refs, on="metric_key", how="left")
df = df.join(target_signal_refs, on="metric_key", how="left")
df = df.withColumn(
    "linked_signal_refs",
    F.array_union(
        F.coalesce(F.col("source_signal_refs"), empty_signal_refs),
        F.coalesce(F.col("target_signal_refs"), empty_signal_refs)
    )
).drop("source_signal_refs", "target_signal_refs")

# 9. Build peer context struct only when a benchmark row exists
df = df.withColumn(
    "peer_context",
    F.when(
        F.col("peer_rank").isNotNull(),
        F.struct(
            "peer_rank",
            "peer_club_count",
            "peer_median",
            "peer_leader_value",
            "gap_to_peer_median",
            "gap_to_leader"
        )
    ).otherwise(F.lit(None))
)

# 10. Generate deterministic candidate id and write output
df = df.withColumn(
    "priority_candidate_id",
    F.concat_ws("_", F.date_format(F.col("month"), "yyyy-MM-dd"), F.col("asset_name"), F.col("metric_name"))
)

final_cols = [
    "month",
    "priority_candidate_id",
    "asset_name",
    "metric_name",
    "metric_value",
    "health_status",
    "trend_direction",
    "deviation_from_rolling_avg",
    "seasonal_z_score",
    "severity_score",
    "persistence_months",
    "persistence_score",
    "peer_gap_score",
    "commercial_weight_score",
    "supporting_evidence_score",
    "peer_context",
    "linked_signal_refs",
    "supporting_metric_rows",
    "category"
]

gold_inputs = df.select(*final_cols)
gold_inputs.write.format("delta").mode("overwrite").saveAsTable("clubos.gold.gold_priority_inputs")
print(f"Computed priority inputs with evidence chain fields: {gold_inputs.count()} rows output.")
