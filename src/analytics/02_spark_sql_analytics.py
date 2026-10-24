"""
========================================================================================
MODULE 2: BATCH ANALYTICS WITH PYSPARK DATAFRAMES & SPARK SQL
File: src/analytics/02_spark_sql_analytics.py
========================================================================================

PURPOSE FOR STUDENTS & ASSIGNMENT DEFENSE:
----------------------------------------------------------------------------------------
1. Demonstrates Spark DataFrames and the Catalyst Query Optimizer.
2. Registers a temporary SQL view (`createOrReplaceTempView`) to run standard ANSI SQL.
3. Showcases SQL aggregations, conditional expressions (CASE WHEN), and Window Functions.
4. Saves analytical summary tables to the `data/processed/` directory.

HOW TO RUN:
----------------------------------------------------------------------------------------
python src/analytics/02_spark_sql_analytics.py
========================================================================================
"""

import os
import sys

# Ensure PySpark workers use the active virtual environment Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Standardize output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, avg, count, when


def main():
    print("=" * 80)
    print("[MODULE 2] PYSPARK DATAFRAMES & SPARK SQL ANALYTICS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: INITIALIZE SPARK SESSION
    # -------------------------------------------------------------------------
    # SparkSession is the unified entry point for DataFrames and Spark SQL (since Spark 2.0)
    spark = (
        SparkSession.builder
        .appName("EcommerceSparkSQLAnalytics")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Locate dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DATA] Loading dataset from: {data_path}")

    # -------------------------------------------------------------------------
    # STEP 2: LOAD CSV INTO PYSPARK DATAFRAME
    # -------------------------------------------------------------------------
    # inferSchema automatically identifies integer, double, boolean types
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(data_path)
    )

    print("[SCHEMA] Inferred DataFrame Schema:")
    df.printSchema()

    total_records = df.count()
    print(f"[RECORDS] Successfully loaded {total_records:,} session records.\n")

    # -------------------------------------------------------------------------
    # STEP 3: REGISTER TEMPORARY SQL VIEW
    # -------------------------------------------------------------------------
    # This enables running ANSI SQL directly against the in-memory Spark DataFrame
    df.createOrReplaceTempView("sessions")

    # -------------------------------------------------------------------------
    # QUERY 1: CONVERSION RATE BY MONTH (SEASONAL TRENDS)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("📈 QUERY 1: E-COMMERCE CONVERSION RATES ACROSS MONTHS")
    print("=" * 80)
    query_month = """
        SELECT 
            Month,
            COUNT(*) AS total_sessions,
            SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions,
            ROUND((SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS conversion_rate_pct,
            ROUND(AVG(PageValues), 2) AS avg_page_value,
            ROUND(AVG(ExitRates) * 100.0, 2) AS avg_exit_rate_pct
        FROM sessions
        GROUP BY Month
        ORDER BY total_sessions DESC
    """
    month_df = spark.sql(query_month)
    month_df.show(truncate=False)

    # -------------------------------------------------------------------------
    # QUERY 2: PURCHASING BEHAVIOR BY VISITOR TYPE
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("👥 QUERY 2: VISITOR SEGMENTATION & PURCHASE PROPENSITY")
    print("=" * 80)
    query_visitor = """
        SELECT 
            VisitorType,
            COUNT(*) AS total_sessions,
            SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions,
            ROUND((SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS conversion_rate_pct,
            ROUND(AVG(ProductRelated_Duration), 2) AS avg_product_seconds,
            ROUND(AVG(BounceRates) * 100.0, 2) AS avg_bounce_pct
        FROM sessions
        GROUP BY VisitorType
        ORDER BY total_sessions DESC
    """
    visitor_df = spark.sql(query_visitor)
    visitor_df.show(truncate=False)

    # -------------------------------------------------------------------------
    # QUERY 3: ADVANCED WINDOW FUNCTION - TRAFFIC SOURCE RANKING
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("🚦 QUERY 3: TRAFFIC SOURCE RANKING USING SQL WINDOW FUNCTIONS")
    print("=" * 80)
    query_traffic = """
        WITH TrafficStats AS (
            SELECT 
                TrafficType,
                COUNT(*) AS total_sessions,
                SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions,
                ROUND((SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS conversion_rate_pct
            FROM sessions
            GROUP BY TrafficType
        )
        SELECT 
            TrafficType,
            total_sessions,
            conversions,
            conversion_rate_pct,
            DENSE_RANK() OVER (ORDER BY conversions DESC) AS sales_volume_rank
        FROM TrafficStats
        ORDER BY sales_volume_rank ASC
        LIMIT 10
    """
    traffic_df = spark.sql(query_traffic)
    traffic_df.show(truncate=False)

    # -------------------------------------------------------------------------
    # QUERY 4: PAGE VALUES & INTENT CORRELATION
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("🎯 QUERY 4: HIGH-INTENT BROWSING COMPARISON (PAGE VALUES)")
    print("=" * 80)
    query_intent = """
        SELECT 
            CASE 
                WHEN PageValues = 0 THEN '0. Zero Page Value (Passive)'
                WHEN PageValues > 0 AND PageValues <= 30 THEN '1. Low-to-Medium Intent (0-30)'
                ELSE '2. High Intent (>30)'
            END AS intent_tier,
            COUNT(*) AS total_sessions,
            SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) AS conversions,
            ROUND((SUM(CASE WHEN Revenue = true THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS conversion_rate_pct
        FROM sessions
        GROUP BY 1
        ORDER BY intent_tier ASC
    """
    intent_df = spark.sql(query_intent)
    intent_df.show(truncate=False)

    # -------------------------------------------------------------------------
    # STEP 4: PERSIST SUMMARY ANALYTICS
    # -------------------------------------------------------------------------
    # Convert month summary to pandas via row dictionaries and save as CSV
    import pandas as pd
    summary_csv_path = os.path.join(output_dir, "batch_monthly_summary.csv")
    summary_rows = [row.asDict() for row in month_df.collect()]
    summary_pdf = pd.DataFrame(summary_rows)
    summary_pdf.to_csv(summary_csv_path, index=False)
    print(f"[SAVE] Exported monthly summary analytics to: {summary_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 5: CLEAN TERMINATION
    # -------------------------------------------------------------------------
    spark.stop()
    print("[SUCCESS] SparkSession stopped successfully.")


if __name__ == "__main__":
    main()
