"""
MODULE 4: REAL-TIME EVENT PRODUCER (KAFKA / STREAM BUFFER)
File: src/producer/kafka_producer.py
"""

import os
import sys
import time
import json
import uuid
import argparse
from datetime import datetime, timezone

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_kafka_producer(bootstrap_servers):
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=3000,
            max_block_ms=3000,
        )
        producer.bootstrap_connected()
        print(f"[KAFKA] Connected to Kafka broker at {bootstrap_servers}")
        return producer
    except Exception:
        print("[PRODUCER] Running with local stream buffer (Kafka offline or in container buffer mode)")
        return None


def main():
    default_bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
    parser = argparse.ArgumentParser(description="E-Commerce Live Session Event Producer")
    parser.add_argument("--rate", type=float, default=0.5, help="Seconds delay between events (default: 0.5)")
    parser.add_argument("--limit", type=int, default=0, help="Maximum events (0 = continuous)")
    parser.add_argument("--bootstrap", type=str, default=default_bootstrap, help="Kafka bootstrap server")
    parser.add_argument("--topic", type=str, default="ecommerce-sessions", help="Kafka topic name")
    args = parser.parse_args()

    print(f"[PRODUCER START] Emitting session events every {args.rate}s...")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_csv = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")
    buffer_file = os.path.join(base_dir, "data", "processed", "live_stream_buffer.jsonl")
    os.makedirs(os.path.dirname(buffer_file), exist_ok=True)

    if not os.path.exists(raw_csv):
        print(f"[ERROR] Source dataset not found: {raw_csv}")
        sys.exit(1)

    producer = get_kafka_producer(args.bootstrap)

    with open(raw_csv, "r", encoding="utf-8") as f:
        header = f.readline()
        rows = [line.strip().split(",") for line in f if line.strip()]

    emitted_count = 0
    row_idx = 0

    try:
        while True:
            row = rows[row_idx]
            row_idx = (row_idx + 1) % len(rows)

            event = {
                "session_id": f"SES-{uuid.uuid4().hex[:8].upper()}",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "Administrative": int(row[0]),
                "Administrative_Duration": float(row[1]),
                "Informational": int(row[2]),
                "Informational_Duration": float(row[3]),
                "ProductRelated": int(row[4]),
                "ProductRelated_Duration": float(row[5]),
                "BounceRates": float(row[6]),
                "ExitRates": float(row[7]),
                "PageValues": float(row[8]),
                "SpecialDay": float(row[9]),
                "Month": row[10].strip(),
                "OperatingSystems": int(row[11]),
                "Browser": int(row[12]),
                "Region": int(row[13]),
                "TrafficType": int(row[14]),
                "VisitorType": row[15].strip(),
                "Weekend": row[16].strip().upper() == "TRUE",
                "Revenue": row[17].strip().upper() == "TRUE",
            }

            if producer is not None:
                try:
                    producer.send(args.topic, value=event)
                except Exception:
                    pass

            with open(buffer_file, "a", encoding="utf-8") as bf:
                bf.write(json.dumps(event) + "\n")

            emitted_count += 1

            # Print summary heartbeat every 20 events to avoid terminal clutter
            if emitted_count % 20 == 0:
                print(f"[PRODUCER HEARTBEAT] Streamed {emitted_count} sessions | Latest: {event['session_id']} ({event['VisitorType']})")

            if args.limit > 0 and emitted_count >= args.limit:
                print(f"[PRODUCER DONE] Reached limit of {args.limit} events.")
                break

            time.sleep(args.rate)

    except KeyboardInterrupt:
        print("\n[PRODUCER STOP] Stopped by user.")
    finally:
        if producer is not None:
            producer.flush()
            producer.close()


if __name__ == "__main__":
    main()
