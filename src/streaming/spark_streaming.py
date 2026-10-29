"""
MODULE 5: SPARK STRUCTURED STREAMING & REAL-TIME ML INFERENCE
File: src/streaming/spark_streaming.py
"""

import os
import sys
import time
import json

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if os.path.exists(r"C:\Spark\bin\winutils.exe"):
    os.environ["HADOOP_HOME"] = r"C:\Spark"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
)
from pyspark.sql.functions import col, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel


def main():
    print("[STREAMING START] Initializing Spark Structured Streaming Engine...")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, "models", "spark_purchase_model")
    buffer_file = os.path.join(base_dir, "data", "processed", "live_stream_buffer.jsonl")
    predictions_file = os.path.join(base_dir, "data", "processed", "live_predictions.jsonl")
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)

    if not os.path.exists(model_dir):
        print(f"[ERROR] Trained PySpark MLlib model not found at: {model_dir}")
        print("Run 'python src/ml/train.py' first.")
        sys.exit(1)

    spark = (
        SparkSession.builder
        .appName("EcommerceRealtimeScoring")
        .master("local[*]")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    model = PipelineModel.load(model_dir)
    print("[STREAMING READY] PySpark MLlib model loaded. Listening for incoming stream...")

    schema = StructType([
        StructField("session_id", StringType(), True),
        StructField("event_timestamp", StringType(), True),
        StructField("Administrative", IntegerType(), True),
        StructField("Administrative_Duration", DoubleType(), True),
        StructField("Informational", IntegerType(), True),
        StructField("Informational_Duration", DoubleType(), True),
        StructField("ProductRelated", IntegerType(), True),
        StructField("ProductRelated_Duration", DoubleType(), True),
        StructField("BounceRates", DoubleType(), True),
        StructField("ExitRates", DoubleType(), True),
        StructField("PageValues", DoubleType(), True),
        StructField("SpecialDay", DoubleType(), True),
        StructField("Month", StringType(), True),
        StructField("OperatingSystems", IntegerType(), True),
        StructField("Browser", IntegerType(), True),
        StructField("Region", IntegerType(), True),
        StructField("TrafficType", IntegerType(), True),
        StructField("VisitorType", StringType(), True),
        StructField("Weekend", BooleanType(), True),
        StructField("Revenue", BooleanType(), True),
    ])

    processed_lines = 0

    try:
        while True:
            if not os.path.exists(buffer_file):
                time.sleep(1)
                continue

            with open(buffer_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            new_lines = all_lines[processed_lines:]
            if not new_lines:
                time.sleep(1)
                continue

            #! Batch process up to 30 events per cycle
            batch_lines = new_lines[:30]
            records = [json.loads(line) for line in batch_lines if line.strip()]
            if records:
                batch_df = spark.createDataFrame(records, schema=schema)
                batch_clean = batch_df.withColumn(
                    "Weekend_num", when(col("Weekend") == True, 1.0).otherwise(0.0)
                )

                # Distributed model scoring
                scored_df = model.transform(batch_clean)
                scored_with_prob = scored_df.withColumn(
                    "purchase_prob", vector_to_array(col("probability"))[1]
                )

                results = scored_with_prob.select(
                    "session_id",
                    "event_timestamp",
                    "VisitorType",
                    "Month",
                    "PageValues",
                    "ExitRates",
                    "prediction",
                    "purchase_prob",
                    "Revenue",
                ).collect()

                high_intent_count = 0
                with open(predictions_file, "a", encoding="utf-8") as pf:
                    for r in results:
                        rec = r.asDict()
                        if rec["prediction"] == 1.0:
                            high_intent_count += 1
                        pf.write(json.dumps(rec) + "\n")

                processed_lines += len(batch_lines)
                print(f"[STREAMING BATCH] Scored {len(results)} sessions | High-Intent (Buy): {high_intent_count} | Total Scored: {processed_lines}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STREAMING STOP] Stopped cleanly.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
