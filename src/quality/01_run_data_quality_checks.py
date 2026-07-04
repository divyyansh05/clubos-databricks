# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality Checks
# MAGIC Runs deterministic validations across Silver tables and fail-stops the run when required checks fail.

import uuid
import pyspark.sql.functions as F

run_id = str(uuid.uuid4())
checks = []
required_failures = []

print(f"Starting Quality Validation Run: {run_id}")


def record_check(table_name: str, check_name: str, severity: str, issue_count: int, issue_details: str) -> None:
    status = "PASS" if issue_count == 0 else "FAIL"
    checks.append({
        "run_id": run_id,
        "table_name": table_name,
        "check_name": check_name,
        "severity": severity,
        "status": status,
        "issue_count": issue_count,
        "issue_details": issue_details,
    })

    print(f"[{status}] [{severity}] {table_name} - {check_name} ({issue_count} issues)")

    if severity == "REQUIRED" and issue_count > 0:
        required_failures.append(f"{table_name} :: {check_name} :: {issue_count} issues")


def run_condition_check(table_name: str, check_name: str, condition_expr: str, severity: str = "REQUIRED") -> None:
    df = spark.read.table(table_name)
    issue_count = df.filter(~F.expr(condition_expr)).count()
    record_check(
        table_name=table_name,
        check_name=check_name,
        severity=severity,
        issue_count=issue_count,
        issue_details=f"Failed condition: {condition_expr}",
    )


def run_duplicate_check(table_name: str, key_columns: list[str], severity: str = "REQUIRED") -> None:
    df = spark.read.table(table_name)
    duplicate_count = df.groupBy(*key_columns).count().filter(F.col("count") > 1).count()
    record_check(
        table_name=table_name,
        check_name=f"No duplicate keys on ({', '.join(key_columns)})",
        severity=severity,
        issue_count=duplicate_count,
        issue_details=f"Duplicate key groups found for columns: {key_columns}",
    )


def run_benchmark_coverage_check(severity: str = "REQUIRED") -> None:
    table_name = "clubos.silver.silver_benchmark_asset_metrics"
    df = spark.read.table(table_name)
    counts = df.groupBy("month", "asset_name", "metric_name").agg(F.countDistinct("club").alias("club_count"))
    issue_count = counts.filter(F.col("club_count") != 5).count()
    record_check(
        table_name=table_name,
        check_name="Exactly 5 benchmark clubs per month+asset+metric",
        severity=severity,
        issue_count=issue_count,
        issue_details="Expected club_count == 5 for each month, asset_name, metric_name.",
    )


def run_month_coverage_check(severity: str = "REQUIRED") -> None:
    table_name = "clubos.silver.silver_internal_asset_metrics"
    df = spark.read.table(table_name)
    month_total = df.select("month").distinct().count()
    per_metric = df.groupBy("asset_name", "metric_name").agg(F.countDistinct("month").alias("month_count"))
    issue_count = per_metric.filter(F.col("month_count") != F.lit(month_total)).count()
    record_check(
        table_name=table_name,
        check_name="Complete monthly coverage by asset+metric",
        severity=severity,
        issue_count=issue_count,
        issue_details=f"Expected each asset+metric to have month_count == {month_total}.",
    )


def run_rate_bounds_check(severity: str = "REQUIRED") -> None:
    table_name = "clubos.silver.silver_internal_asset_metrics"
    df = spark.read.table(table_name)
    rate_rows = df.filter(F.col("metric_name").rlike("(rate$|_rate$|recurrence$|bounce_rate$)"))
    issue_count = rate_rows.filter((F.col("metric_value") < 0) | (F.col("metric_value") > 1)).count()
    record_check(
        table_name=table_name,
        check_name="Rate and recurrence metrics bounded between 0 and 1",
        severity=severity,
        issue_count=issue_count,
        issue_details="Found rate-like metric_value outside [0,1].",
    )


# Required checks
run_condition_check("clubos.silver.silver_internal_asset_metrics", "No null months", "month IS NOT NULL")
run_condition_check("clubos.silver.silver_internal_asset_metrics", "No null metric values", "metric_value IS NOT NULL")
run_duplicate_check("clubos.silver.silver_internal_asset_metrics", ["month", "asset_name", "metric_name"])
run_duplicate_check("clubos.silver.silver_benchmark_asset_metrics", ["month", "asset_name", "metric_name", "club"])
run_month_coverage_check()
run_benchmark_coverage_check()
run_rate_bounds_check()

# Persist run log before raising fail-stop
log_df = spark.createDataFrame(checks).withColumn("run_timestamp", F.current_timestamp())
log_df.write.format("delta").mode("append").saveAsTable("clubos.silver.silver_data_quality_checks")

if len(required_failures) > 0:
    failure_message = "Quality gate failed. Required checks did not pass:\n- " + "\n- ".join(required_failures)
    raise RuntimeError(failure_message)

print("Data quality checks complete. All required checks passed.")
