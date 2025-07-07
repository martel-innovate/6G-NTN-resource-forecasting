from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta, timezone
import psycopg2
from prefect import flow, task
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_NAME=os.getenv('DB_NAME')
DB_USER=os.getenv('DB_USER')
DB_PASSWORD=os.getenv('DB_SECRET')
DB_HOST=os.getenv('DB_HOSTNAME')
DB_PORT=os.getenv('DB_PORT')
PROMETHEUS_HOSTNAME = os.getenv('PROMETHEUS_HOSTNAME')
#METRIC = "container_cpu_usage_seconds_total" 
METRIC = "container_memory_usage_bytes"

@task
def fetch_cpu_usage():
    # Initialize Prometheus client
    print(f"Connecting to Prometheus at http://{PROMETHEUS_HOSTNAME}")
    prom = PrometheusConnect(url=F'http://{PROMETHEUS_HOSTNAME}', disable_ssl=True)
    #query = ('rate(container_cpu_usage_seconds_total{pod="alertmanager-prometheus-kube-prometheus-alertmanager-0"}[1m])')
    query =  (f'{METRIC}{{node="6g-ntn-f5gc-w1", container="upf1"}}')
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=10)
    try:
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='5s'
        )
        print(f"Fetched {len(result)} data points from Prometheus.")
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

@task
def transform_data(result):
    transformed = []
    for entry in result:
        metric_name = entry['metric'].get('__name__')  # e.g., 'container_memory_usage_bytes'
        container_name = entry['metric'].get('container')  # e.g., 'upf1'

        # Extract base metric type from the full Prometheus metric name
        if metric_name and container_name:
            if 'memory' in metric_name:
                metric_type = 'memory_usage'
            elif 'cpu' in metric_name:
                metric_type = 'cpu_usage'
            else:
                metric_type = metric_name  # fallback to full name

        # Build variable name
        name_in_db = f"{metric_type}_{container_name}"
        for value in entry['values']:
            timestamp = datetime.fromtimestamp(float(value[0]), tz=timezone.utc)
            cpu_value = float(value[1])
            transformed.append((name_in_db, cpu_value, timestamp))
    return transformed

@task
def insert_to_db(data):
    print(f"Inserting {len(data)} records into the database.")
    print(f"Connecting to database {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        insert_query = "INSERT INTO input (metric_name, value, datetime) VALUES (%s, %s, %s);"
        cursor.executemany(insert_query, data)
        conn.commit()
        cursor.close()
        conn.close()
        print("Data successfully inserted.")
    except Exception as e:
        print(f"Database error: {e}")

@flow
def prometheus_to_postgres():
    raw_data = fetch_cpu_usage()
    transformed_data = transform_data(raw_data)
    insert_to_db(transformed_data)

if __name__ == '__main__':
    prometheus_to_postgres()
