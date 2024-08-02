import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from prefect import flow, task

from dotenv import load_dotenv

from prometheus_client.client import PrometheusClient
from prometheus_client.relevant_params import relevant_metrics_names,relevant_nfs

# load env variables
load_dotenv()

DB_SECRET = os.getenv('DB_SECRET')
DB_HOSTNAME = os.getenv('DB_HOSTNAME')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')

#### 

# This is a script that can be used to upload the content of csv files to forecasting postgres database

####


@task(log_prints=True)
def read_from_csv(directory_path):
    # set pattern Prometheus file
    start_string_cpu = "metric_container_cpu_usage_seconds_total_"
    start_string_memory = "metric_container_memory_usage_bytes_"

    # Iterate over each CSV file in the directory
    for filename in os.listdir(directory_path):
        if filename.startswith("metric") and filename.endswith(".csv"):
            # Read CSV file into a DataFrame and append to list
            df = pd.read_csv(os.path.join(directory_path, filename))

            if filename.startswith(start_string_cpu):
                # Create column name
                column_name = "cpu_usage_" + filename.rstrip(".csv")[-3:]
            elif filename.startswith(start_string_memory):
                # Create column name
                column_name = "memory_usage_" + filename.rstrip(".csv")[-3:]

            df["metric_name"] = column_name
            df.rename(columns={'time': 'datetime'}, inplace=True)

            # convert timestamp in datetime with timezone
            datetime_timestamp = pd.to_datetime(df['datetime'], unit='s', origin="unix", utc="True")
            df['datetime'] = datetime_timestamp.dt.floor('S')
    return df

def get_session():
    # Define database connection parameters
    DATABASE_URI = f'postgresql+psycopg2://{DB_USER}:{DB_SECRET}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}'

    # Create an engine
    engine = create_engine(DATABASE_URI)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a Session
    session = Session()

    return session

@task(log_prints=True)
def load_df_to_postgres(logger, df):
    # connect to postres
    session = get_session()   

    logger.info(df.columns) 
    
    df["metric_name"]=df["metric"]
    try:
        # Load data into input table
        for index, row in df.iterrows():
            # Use parameterized queries to prevent SQL injection
            query = text("INSERT INTO input (metric_name, datetime, value) VALUES (:metric_name, :datetime, :value)")
            params = {
                'metric_name': row['metric_name'],
                'datetime': row['datetime'],
                'value': row['value']
            }
            session.execute(query, params)

        # Commit the transaction after the loop
        session.commit()
        logger.info("Data inserted successfully.")

    except SQLAlchemyError as e:
        # Rollback the transaction in case of error
        session.rollback()
        logger.error(f"An error occurred: {e}")

    finally:
        # Close the session
        session.close()
    
    logger.info("loaded successfully!")
    return

# A dictionary to match the original metric names with those expected in the time series.
map_metrics_names={
 "container_cpu_usage_seconds_total":"cpu_usage",
 "container_memory_usage_bytes":"memory_usage",
 "open5gs-amf":"amf",
 "open5gs-pcf":"pcf",
 "open5gs-ausf":"ausf",
 "open5gs-smf":"smf"
}

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

class PrometheusDataProcessor:
    """ Processes raw Prometheus data into structured DataFrames. """

    @staticmethod
    def process_data(raw_data):
        if raw_data is None:
            return None
        filtered_data = [
            {"metric": map_metrics_names.get(item["metric"]["__name__"]) + "_" + map_metrics_names.get("-".join(item["metric"]["pod"].split("-")[:2])),
             "datetime": item["value"][0], "value": item["value"][1]}
            for item in raw_data
        ]
        return pd.DataFrame(filtered_data)
    
class PrometheusRangeDataProcessor():
    """ Processes raw Prometheus range data into structured DataFrames. """
    @staticmethod
    def process_data(raw_data):
        if raw_data is None:
            return None
        filtered_data = [
            {"metric": map_metrics_names.get(item["metric"]["__name__"]) + "_" + map_metrics_names.get("-".join(item["metric"]["pod"].split("-")[:2])),
             "values":item["values"]}
            for item in raw_data
        ]

        metrics = []
        datetimes = []
        values = []
        for item in filtered_data:
            metric = item["metric"]
            for value_pair in item["values"]:
                datetime, value = value_pair
                metrics.append(metric)
                datetimes.append(datetime)
                values.append(value)
        df=pd.DataFrame({"metric": metrics, "datetime": datetimes, "value": values})
        df["datetime"]=df["datetime"].astype(int)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        return df

@task(log_prints=True)
def prom_extract_and_transform():
    #extracts metrics data from prometheus calling the api, filters the metrics that are relevant, calls process_data to transform it into structuresd pandas dataframe 
    metrics_list=[]
    pc=PrometheusClient()
    relevant_metrics=flatten_list([pc.fetch_relevant_metrics(x) for x in relevant_metrics_names ])
    for m in relevant_metrics:
        res=pc.intent_query(m,time_range="5m").json()["data"].get("result")
        for nf in relevant_nfs:
            filtered_list=list(filter(
                    lambda a: "pod" in a["metric"] and a["metric"].get("pod").startswith(nf) ,
                    res
                ))
            metrics_list.extend(filtered_list)
    #processed_data = PrometheusDataProcessor.process_data(metrics_list)
    processed_data = PrometheusRangeDataProcessor.process_data(metrics_list)
    return processed_data

@flow(log_prints=True)
def get_input():

    # get logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # get metrics with Prometheus API
    df = prom_extract_and_transform()

    # load data into postgres
    load_df_to_postgres(logger, df)

    return


if __name__ == "__main__":
    get_input()