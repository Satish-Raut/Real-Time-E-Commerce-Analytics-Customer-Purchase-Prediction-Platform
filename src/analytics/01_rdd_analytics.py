"""
========================================================================================
MODULE 1: FOUNDATIONAL DISTRIBUTED ANALYTICS WITH PYSPARK RDDs
File: src/analytics/01_rdd_analytics.py
========================================================================================

WHY THIS SCRIPT IS WRITTEN THIS WAY (CLI vs. Python Scripting Guide):
----------------------------------------------------------------------------------------
1. In the interactive terminal (`pyspark` command in cmd), Spark automatically creates
   and injects a global variable named `sc` (SparkContext) and `spark` (SparkSession).
2. In a Python script or IDE (like VS Code / PyCharm / Antigravity), there is NO pre-existing
   `sc`. You must explicitly:
     - Import `SparkConf` and `SparkContext` from `pyspark`.
     - Configure memory, master URL (`local[*]`), and app name.
     - Instantiate `SparkContext(conf=conf)`.
     - Always call `sc.stop()` at the end to release JVM and cluster resources.

EDUCATIONAL OBJECTIVES DEMONSTRATED IN THIS SCRIPT:
----------------------------------------------------------------------------------------
1. Low-Level Distributed Primitives:
   - RDD Creation: `sc.textFile()` reads distributed file partitions.
   - Lineage Graph (DAG) & Lazy Evaluation: Transformations do not execute until an Action.
   - Header Extraction: `.first()` (Action) and `.filter()` (Transformation).
2. Key RDD Transformations:
   - `.map()`: Parse delimited text lines into structured tuples.
   - `.filter()`: Distributed boolean filtering (e.g., high-intent sessions).
   - `.mapValues()`: Transform values without shuffling keys.
   - `.reduceByKey()`: Highly efficient map-side combiners to aggregate metrics by key.
   - `.sortBy()`: Distributed sort across cluster partitions.
3. RDD Actions:
   - `.count()`, `.take()`, `.collect()`.
========================================================================================
"""

import os
import sys

# Critical for Windows environments:
# Force PySpark worker processes to use the exact Python executable from the active
# virtual environment, avoiding Windows Store 'python not found' execution aliases.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark import SparkConf, SparkContext


def parse_csv_line(line: str):
    """
    Parses a single line of CSV text safely.
    Handles potential formatting issues and maps to structured fields.

    Dataset Schema (18 fields):
    0: Administrative (int)
    1: Administrative_Duration (float)
    2: Informational (int)
    3: Informational_Duration (float)
    4: ProductRelated (int)
    5: ProductRelated_Duration (float)
    6: BounceRates (float)
    7: ExitRates (float)
    8: PageValues (float)
    9: SpecialDay (float)
    10: Month (str)
    11: OperatingSystems (int)
    12: Browser (int)
    13: Region (int)
    14: TrafficType (int)
    15: VisitorType (str)
    16: Weekend (bool: TRUE/FALSE)
    17: Revenue (bool: TRUE/FALSE - Target Label)
    """
    parts = line.strip().split(",")
    if len(parts) < 18:
        return None

    try:
        return {
            "administrative": int(parts[0]),
            "admin_duration": float(parts[1]),
            "informational": int(parts[2]),
            "info_duration": float(parts[3]),
            "product_related": int(parts[4]),
            "product_duration": float(parts[5]),
            "bounce_rates": float(parts[6]),
            "exit_rates": float(parts[7]),
            "page_values": float(parts[8]),
            "special_day": float(parts[9]),
            "month": parts[10].strip(),
            "operating_systems": int(parts[11]),
            "browser": int(parts[12]),
            "region": int(parts[13]),
            "traffic_type": int(parts[14]),
            "visitor_type": parts[15].strip(),
            "weekend": parts[16].strip().upper() == "TRUE",
            "revenue": parts[17].strip().upper() == "TRUE",
        }
    except (ValueError, IndexError):
        return None


# Reconfigure standard output to UTF-8 on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 80)
    print("[JOB START] STARTING PYSPARK RDD DISTRIBUTED ANALYTICS JOB")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: INITIALIZE SPARK CONFIGURATION & SPARKCONTEXT
    # -------------------------------------------------------------------------
    # When running outside the interactive CLI, we must construct the SparkConf:
    # - setMaster("local[*]"): Uses all available CPU cores on this local machine.
    # - setAppName("EcommerceRDDAnalytics"): Descriptive name in Spark UI.
    conf = (
        SparkConf()
        .setAppName("EcommerceRDDAnalytics")
        .setMaster("local[*]")
        .set("spark.ui.showConsoleProgress", "true")
    )

    # Create the SparkContext (the entry point for low-level RDD programming)
    sc = SparkContext.getOrCreate(conf=conf)
    # Set log level to WARN to reduce verbose INFO logs in terminal
    sc.setLogLevel("WARN")

    # -------------------------------------------------------------------------
    # STEP 2: LOCATE & LOAD RAW CSV DATA AS RDD
    # -------------------------------------------------------------------------
    # Build a robust path relative to the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")

    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset file not found at: {data_path}")
        sc.stop()
        sys.exit(1)

    print(f"[DATA] Loading raw dataset from: {data_path}")

    # textFile() reads the file into an RDD of raw text lines across partitions
    raw_lines_rdd = sc.textFile(data_path)

    # -------------------------------------------------------------------------
    # STEP 3: HEADER EXTRACTION & ROW FILTERING
    # -------------------------------------------------------------------------
    # Action: .first() retrieves the very first line from partition 0 (the header)
    header_line = raw_lines_rdd.first()
    print(f"[HEADER] Detected CSV Header: {header_line}")

    # Transformation: .filter() lazily skips the header line from all partitions
    data_lines_rdd = raw_lines_rdd.filter(lambda line: line != header_line)

    # Transformation: .map() parses each line into our Python dictionary record
    parsed_sessions_rdd = data_lines_rdd.map(parse_csv_line).filter(lambda record: record is not None)

    # Cache the parsed RDD in memory since we will run multiple analytics on it
    parsed_sessions_rdd.cache()

    # Action: .count() triggers computation and returns total records
    total_sessions = parsed_sessions_rdd.count()
    print(f"[LOADED] Total Valid Sessions: {total_sessions:,}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # ANALYSIS 1: OVERALL CONVERSION METRICS (map & filter)
    # -------------------------------------------------------------------------
    # Filter sessions where the user completed a purchase (revenue == True)
    conversions_rdd = parsed_sessions_rdd.filter(lambda s: s["revenue"] is True)
    total_conversions = conversions_rdd.count()
    conversion_rate = (total_conversions / total_sessions) * 100

    print("--- 1. OVERALL CONVERSION METRICS ---")
    print(f"   - Total Purchasing Sessions (Revenue=True): {total_conversions:,}")
    print(f"   - Total Non-Purchasing Sessions:           {total_sessions - total_conversions:,}")
    print(f"   - Global Benchmark Conversion Rate:        {conversion_rate:.2f}%")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # ANALYSIS 2: CONVERSION RATE BY VISITOR TYPE (reduceByKey & mapValues)
    # -------------------------------------------------------------------------
    # Pair RDD Pattern:
    # Key   : visitor_type (e.g., 'Returning_Visitor', 'New_Visitor', 'Other')
    # Value : (total_sessions_increment, converted_sessions_increment)
    visitor_pairs_rdd = parsed_sessions_rdd.map(
        lambda s: (s["visitor_type"], (1, 1 if s["revenue"] else 0))
    )

    # reduceByKey() uses map-side combine to aggregate tuples per partition
    # before shuffling across the cluster (much faster than groupByKey!)
    visitor_aggregated_rdd = visitor_pairs_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    # mapValues() computes the final conversion rate without modifying the key
    visitor_metrics_rdd = visitor_aggregated_rdd.mapValues(
        lambda agg: {
            "total_sessions": agg[0],
            "conversions": agg[1],
            "conversion_rate_pct": round((agg[1] / agg[0]) * 100, 2),
        }
    )

    # Action: .collect() brings the aggregated summary to the driver
    print("--- 2. CONVERSION METRICS BY VISITOR TYPE (reduceByKey) ---")
    print(f"   {'Visitor Type':<22} | {'Sessions':<10} | {'Purchases':<10} | {'Conversion %':<12}")
    print("   " + "-" * 62)
    for vtype, stats in visitor_metrics_rdd.collect():
        print(
            f"   {vtype:<22} | {stats['total_sessions']:<10,d} | {stats['conversions']:<10,d} | {stats['conversion_rate_pct']:>6.2f}%"
        )
    print("-" * 80)

    # -------------------------------------------------------------------------
    # ANALYSIS 3: BOUNCE & EXIT RATES BY MONTH (aggregate metrics)
    # -------------------------------------------------------------------------
    # Key   : Month (e.g. 'May', 'Nov', 'Mar')
    # Value : (bounce_rate, exit_rate, count=1)
    month_rates_rdd = parsed_sessions_rdd.map(
        lambda s: (s["month"], (s["bounce_rates"], s["exit_rates"], 1))
    )

    # Sum up bounce rates, exit rates, and session count per month
    month_totals_rdd = month_rates_rdd.reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])
    )

    # Calculate average bounce rate and average exit rate
    month_avg_rdd = month_totals_rdd.mapValues(
        lambda t: {
            "avg_bounce_rate": round((t[0] / t[2]) * 100, 2),
            "avg_exit_rate": round((t[1] / t[2]) * 100, 2),
            "total_sessions": t[2],
        }
    )

    # Sort descending by total sessions
    sorted_months = month_avg_rdd.sortBy(lambda item: item[1]["total_sessions"], ascending=False).collect()

    print("--- 3. ENGAGEMENT METRICS BY MONTH (Top Activity) ---")
    print(f"   {'Month':<8} | {'Total Sessions':<16} | {'Avg Bounce Rate':<17} | {'Avg Exit Rate':<15}")
    print("   " + "-" * 62)
    for month, mstats in sorted_months:
        print(
            f"   {month:<8} | {mstats['total_sessions']:<16,d} | {mstats['avg_bounce_rate']:>6.2f}%          | {mstats['avg_exit_rate']:>6.2f}%"
        )
    print("-" * 80)

    # -------------------------------------------------------------------------
    # ANALYSIS 4: HIGH-INTENT SESSIONS (PageValues > 50) FILTER & CONVERSION
    # -------------------------------------------------------------------------
    # PageValues indicates the monetary worth of pages visited during a session.
    # We filter high-intent sessions (PageValues > 50) and inspect their conversion.
    high_intent_rdd = parsed_sessions_rdd.filter(lambda s: s["page_values"] > 50.0)
    high_intent_count = high_intent_rdd.count()
    high_intent_purchases = high_intent_rdd.filter(lambda s: s["revenue"] is True).count()
    high_intent_rate = (high_intent_purchases / high_intent_count * 100) if high_intent_count > 0 else 0.0

    print("--- 4. HIGH-INTENT BROWSING ANALYSIS (PageValues > 50) ---")
    print(f"   - High-Intent Sessions Detected:       {high_intent_count:,}")
    print(f"   - High-Intent Conversions:             {high_intent_purchases:,}")
    print(f"   - High-Intent Conversion Rate:         {high_intent_rate:.2f}%")
    print(f"   [INSIGHT] High PageValue sessions convert {high_intent_rate / conversion_rate:.1f}x higher than baseline!")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CLEANUP: STOP SPARKCONTEXT
    # -------------------------------------------------------------------------
    # Crucial in scripts: frees Spark JVM threads, ports, and memory.
    sc.stop()
    print("[SUCCESS] SparkContext successfully terminated.")


if __name__ == "__main__":
    main()
