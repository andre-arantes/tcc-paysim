# src/streaming/kafka_producer.py
import csv
import json
import time
import argparse
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_latency = 0.0
total_messages = 0



def create_producer(server):
    """Cria um produtor Kafka com lógica de retentativa."""
    retries = 5
    for i in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=server,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Produtor Kafka criado com sucesso.")
            return producer
        except NoBrokersAvailable:
            print(f"Tentativa {i+1}/{retries}: Broker Kafka indisponível em {server}. Tentando novamente em 5 segundos...")
            time.sleep(5)
    print("Erro: Não foi possível conectar ao broker Kafka após várias tentativas.")
    return None

def stream_transactions(producer, topic, file_path, interval):
    """Lê transações do CSV e as envia para o Kafka."""
    print(f"Iniciando stream de '{file_path}' para o tópico Kafka '{topic}'...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                
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
                        "event_timestamp": time.time() 
                    }
                    producer.send(topic, message)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Enviadas {i + 1} mensagens...")

                    time.sleep(interval)
                except (ValueError, KeyError) as e:
                    print(f"Pulando linha {i+1} devido a erro: {e}")

        producer.flush()
        print("Finalizado o envio de todas as mensagens.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

def main():
    """Função principal para parsear argumentos e iniciar o stream."""
    parser = argparse.ArgumentParser(description="Produtor Kafka para simular transações do PaySim.")
    # Caminho do arquivo corrigido para a estrutura do projeto
    parser.add_argument('--file', default='data/processed/test_data_csv_final/part-00000-c3da4479-6617-4d6d-b266-32a055a685b8-c000.csv', help='Caminho para o arquivo CSV de entrada.')
    parser.add_argument('--topic', default='transactions', help='Tópico Kafka para enviar as mensagens.')
    parser.add_argument('--server', default='localhost:9092', help='Endereço do servidor Kafka.')
    parser.add_argument('--interval', type=float, default=0.01, help='Intervalo em segundos entre mensagens.')

    args = parser.parse_args()
    
    producer = create_producer(args.server)
    if producer:
        stream_transactions(producer, args.topic, args.file, args.interval)

if __name__ == "__main__":
    main()