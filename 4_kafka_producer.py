# File: 4_kafka_producer.py
import csv
import json
import time
from kafka import KafkaProducer

KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "localhost:9092"
CSV_FILE_PATH = 'paysim.csv'

def create_producer():
    """Creates a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        print(f"Error creating Kafka producer: {e}")
        return None

def stream_transactions():
    """Reads transactions from CSV and sends them to Kafka."""
    producer = create_producer()
    if not producer:
        return

    print(f"Streaming transactions from '{CSV_FILE_PATH}' to Kafka topic '{KAFKA_TOPIC}'...")
    with open(CSV_FILE_PATH, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numerical strings to actual numbers for correct schema inference in Spark
            try:
                message = {
                    "step": int(row['step']),
                    "type": row['type'],
                    "amount": float(row['amount']),
                    "nameOrig": row['nameOrig'],
                    "oldbalanceOrg": float(row['oldbalanceOrg']),
                    "newbalanceOrig": float(row['newbalanceOrig']),
                    "nameDest": row['nameDest'],
                    "oldbalanceDest": float(row['oldbalanceDest']),
                    "newbalanceDest": float(row['newbalanceDest']),
                    "isFraud": int(row['isFraud']),
                    "isFlaggedFraud": int(row['isFlaggedFraud']),
                }
                producer.send(KAFKA_TOPIC, message)
                print(f"Sent: {message['type']} - {message['amount']}")
                time.sleep(0.5)  # Simulate a real-time stream
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e} - Row: {row}")

    producer.flush()
    print("Finished streaming.")

if __name__ == "__main__":
    stream_transactions()