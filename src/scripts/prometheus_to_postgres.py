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

@task
def fetch_prometheus_data(metric):
    # Initialize Prometheus client
    print(f"Connecting to Prometheus at http://{PROMETHEUS_HOSTNAME}")
    prom = PrometheusConnect(url=F'http://{PROMETHEUS_HOSTNAME}', disable_ssl=True)
    # PromQL query to calculate per-second CPU usage rate for all UPF pods in the 'open5gs' namespace
    # Aggregates the usage across all containers and CPU cores, returning one series per pod
    query =  (f'sum by(pod, namespace)(rate({metric}{{namespace="open5gs", node="6g-ntn-f5gc-w2", pod=~"upf2-open5gs-upf-.*"}}[1m]))') 

    # Current time (rounded down to full minute)
    now = datetime.now(timezone.utc)
    current_minute = now.replace(second=0, microsecond=0)

    # We want the *previous* full minute
    start_time = current_minute - timedelta(seconds=45)
    end_time = current_minute

    step = '15s'  # 15-second intervals

    print(f"Fetching data from {start_time} to {end_time} every {step}")
    try:
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        print(f"Fetched {len(result)} data points from Prometheus.")
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

@task
def transform_data(result, metric_name):
    transformed = []
    for entry in result:
        pod_name = entry['metric'].get('pod')

        # Extract VNF name from pod name
        if pod_name:
            if 'upf' in pod_name:
                vnf_name = 'upf'
            else:
                raise ValueError(f"Unrecognized VNF in pod name: {pod_name}")

        # Extract base metric type from the full Prometheus metric name
        if metric_name:
            if 'memory' in metric_name:
                metric_type = 'memory_usage'
            elif 'cpu' in metric_name:
                metric_type = 'cpu_usage'
            else:
                raise ValueError(f"Unrecognized metric_name: {metric_name}")

        # Build variable name
        name_in_db = f"{metric_type}_{vnf_name}"

        # Transform each (timestamp, value) pair
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
def prometheus_to_postgres(metric: str = "container_cpu_usage_seconds_total"):
    """Collects Prometheus metric (e.g. CPU or memory) and stores it in DB."""
    raw_data = fetch_prometheus_data(metric)
    transformed_data = transform_data(raw_data, metric)
    insert_to_db(transformed_data)

if __name__ == '__main__':
    prometheus_to_postgres()
