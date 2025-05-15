from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta, timezone
import psycopg2
from prefect import flow, task
import os

@task
def fetch_cpu_usage():
    # Initialize Prometheus client
    prom = PrometheusConnect(url='http://192.168.49.2:30090', disable_ssl=True)
    query = ('rate(container_cpu_usage_seconds_total{pod="alertmanager-prometheus-kube-prometheus-alertmanager-0"}[1m])')
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=10)
    try:
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='60s'
        )
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

@task
def transform_data(result):
    transformed = []
    for entry in result:
        metric_name = entry['metric'].get('__name__', 'cpu_usage')
        for value in entry['values']:
            timestamp = datetime.fromtimestamp(float(value[0]), tz=timezone.utc)
            cpu_value = float(value[1])
            transformed.append((metric_name, cpu_value, timestamp))
    return transformed

@task
def insert_to_db(data):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_SECRET'),
            host=os.getenv('DB_HOSTNAME'),
            port=os.getenv('DB_PORT')
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
