from prometheus_client import start_http_server, Gauge, Counter
import pandas as pd
import time
from itertools import cycle


relevant_nfs=["open5gs-amf","open5gs-smf","open5gs-ausf","open5gs-pcf"]
relevant_metrics_names=["container_cpu_usage_seconds_total","container_memory_usage_bytes"]
pool_nfs= cycle(relevant_nfs)

df = pd.read_csv("amf_cpu_ram.csv")

# Define Prometheus metrics
metrics = {}

# Prometheus metrics initialization
cpu_usage_gauge = Gauge('container_cpu_usage_seconds_total', 'CPU usage over time',['pod'])
ram_usage_gauge = Gauge('container_memory_usage_bytes', 'RAM usage over time',['pod'])

def set_metrics_group(lst):
    for a in lst:
        current_nfs=next(pool_nfs)
        #print(a[0],a[1])
        # Updating Prometheus metrics
        cpu_usage_gauge.labels(pod=current_nfs).set(a[0])
        ram_usage_gauge.labels(pod=current_nfs).set(a[1])
    # Simulating real-time metrics with a delay
    time.sleep(1)

def update_metrics(df):
    '''
    for index, row in df.iterrows():
        timestamp = row['time']
        cpu_usage = row['cpu_usage']
        ram_usage = row['ram_usage']
    '''
    for i in range(0, len(df), len(relevant_nfs)):
        lst=[]
        batch = df.iloc[i: i + len(relevant_nfs)]
        for index, row in batch.iterrows():
           lst.append((row['cpu_usage']  , row['ram_usage']))
        set_metrics_group(lst)

    '''
     # Updating Prometheus metrics
    cpu_usage_gauge.set(cpu_usage)
    ram_usage_gauge.set(ram_usage)
        # Simulating real-time metrics with a delay
    time.sleep(1)
    '''

if __name__ == '__main__':
    # Start the HTTP server to expose metrics
    start_http_server(8000)
    print("Serving metrics on http://localhost:8000/metrics")

    # Keep the script running to serve metrics
    while True:
        # Update Prometheus metrics with data from the DataFrame (no actual update for static data)
        update_metrics(df)
        # Sleep to reduce CPU usage
        #time.sleep(60)