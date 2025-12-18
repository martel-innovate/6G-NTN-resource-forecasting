from prefect import flow
import pandas as pd
import os
import numpy as np
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
import requests

import pytz
from datetime import datetime

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from dotenv import load_dotenv

# load env variables
load_dotenv()

DB_SECRET = os.getenv('DB_SECRET')
DB_HOSTNAME = os.getenv('DB_HOSTNAME')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
ORCHESTRATOR_URL = os.getenv('ORCHESTRATOR_URL')

logger = None

@task()
def device_check():
    # -------- Device Detection --------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        force_float32 = True
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        force_float32 = False
    else:
        device = torch.device("cpu")
        force_float32 = False

    logger.info(f"Using device: {device}, force_float32={force_float32}")
    return force_float32

@task
def load_data(metric_name):
    logger.info("Starting data loading from database")
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_SECRET}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}')
    conn = engine.connect()
    query = text(f'SELECT * FROM input WHERE metric_name = \'{metric_name}\'')
    results = conn.execute(query)
    results_list = results.fetchall()
    logger.info(f"Fetched {len(results_list)} records from the database for metric: {metric_name}")
    df = pd.DataFrame(results_list)
    return df

@task
def data_transformation(df, frequency): 
    # create one column for each metric
    df_pivot = df.pivot_table(index='datetime', columns='metric_name', values='value')

    # Preprocessing
    # frequency is passed as argument
    df_resampled = df_pivot.reset_index().resample(frequency, on="datetime").max().interpolate()
    df_resampled.index = df_resampled.index.tz_localize(None)
    return df_resampled

@task
def normalize_series_production(series):
    # Normalize the time series during production (no split into train and test)
    # Create one single TimeSeries (NO SPLIT)
    if "datetime" in series:
        series = TimeSeries.from_dataframe(series, "datetime")
    else:
        series = TimeSeries.from_dataframe(series.reset_index(), "datetime")

    transformer = Scaler()
    series_transformed = transformer.fit_transform(series)
    return series_transformed, transformer

@task
def preprocessing(df, frequency):
    logger.info(f"Starting preprocessing, frequency={frequency}")
    # data transformation
    df_resampled = data_transformation(df, frequency)
    
    # data normalization
    series_transformed, transformer = normalize_series_production(df_resampled)

    logger.info("Preprocessing completed")
    return series_transformed, transformer


@task
def model_training_production(model_name, series_transformed, force_float32):
    # no need to split train and val, use all data for training
    logger.info("Starting model training")

    if force_float32:  
        series_transformed = series_transformed.astype(np.float32)
        logger.info("Default torch dtype set to float32")

    # define early stopping parameters
    my_stopper = EarlyStopping(
        monitor="train_loss",
        patience=10,
        min_delta=0.0005,
        mode="min",
    )

    my_checkpoint = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1 # Keeps only the best model according to train_loss
    )

    pl_trainer_kwargs = {"callbacks": [my_stopper, my_checkpoint]}

    # build model
    my_model = RNNModel(
        model="LSTM",
        hidden_dim=50,
        dropout=0.0,
        batch_size=8,
        n_epochs=50,
        optimizer_kwargs={"lr": 1e-3},
        model_name=model_name,
        log_tensorboard=False,
        random_state=42,
        training_length=12,
        input_chunk_length=6,
        output_chunk_length=1,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    # train model
    my_model.fit(series_transformed, verbose=True)
    
    # pick best model
    best_model = RNNModel.load_from_checkpoint(model_name=model_name, best=True)
    
    logger.info("Model training completed")
    return best_model

@task
def inference(my_model, target_name, transformer, steps: int = 1):
    # model save_predictions
    logger.info("Starting model save_predictions")
    predictions = my_model.predict(n=steps)

    # inverse transform
    predictions = transformer.inverse_transform(predictions)
    logger.info("Inverse transform applied")

    # create dataframes
    predictions_df = predictions.pd_dataframe()

    # set indexes
    predictions_df_new = predictions_df.reset_index()
    predictions_df_new.index = [target_name]

    # print dataframe
    logger.info(f"Predictions DataFrame:\n{predictions_df_new}")
    return predictions_df_new

@task
def load_to_postgres(predictions):
    # Melt the DataFrame
    df_melted = pd.melt(predictions, id_vars=['datetime'], value_vars=predictions.columns.drop(["datetime"]), var_name='metric_name', value_name='value')

    # Drop the 'Unnamed: 0' column if necessary
    df_melted.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Replace negative values in 'value' column with 0
    df_melted['value'] = df_melted['value'].apply(lambda x: max(x, 0))

    # to delete: transform datetime in current datetim
    # Get current datetime with Europe timezone
    europe_tz = pytz.timezone('Europe/Rome') 
    current_datetime_europe = datetime.now(europe_tz)

    # Update 'datetime' column with the current datetime and timezone
    df_melted['datetime'] = current_datetime_europe

    # connect to db
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_SECRET}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}')

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a Session
    session = Session()

    try:
        # Load data into input table
        for index, row in df_melted.iterrows():
            # Use parameterized queries to prevent SQL injection
            query = text("INSERT INTO forecasted (metric_name, datetime, value) VALUES (:metric_name, :datetime, :value)")
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
    
    return

@task
def save_predictions_as_artifact(formatted_predictions):
    
    # If final_format returns a single dict, wrap it in a list for Prefect artifact
    if isinstance(formatted_predictions, dict):
        formatted_predictions = [formatted_predictions]

    logger.info(f"Preparing Artifact:\n{formatted_predictions}")

    # Save predictions to Prefect Artifacts
    create_table_artifact(
        key="predictions",
        table=formatted_predictions,
        description="The output of Machine Learning in final format"
    )

    logger.info("Predictions saved as Prefect Artifact")
    return formatted_predictions


@task
def final_format(final_predictions, metric_name, target_name, node):
    # Convert JSON data to pandas DataFrame
    df = pd.DataFrame(final_predictions)
    logger.info(f"Initial predictions DataFrame:\n{df}")

    # Keep only required fields
    df = df[['datetime', metric_name]].copy()

    # Rename columns for consistency with the desired output
    df.columns = ['datetime', 'value']

    # Add required column
    df['network_function'] = metric_name[-3:]
    df['metric_name'] = target_name        
    df['node'] = node                     

    # Drop rows where values are NaN
    df.dropna(subset=['value'], how='all', inplace=True)

    # Convert datetime to ISO string
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize("Europe/Rome").apply(lambda x: x.isoformat())

    # Convert to list of dicts
    result = df.to_dict(orient='records')

    logger.info(f"Final formatting of predictions:\n{result}")

    return result

@task
def post_predictions(formatted_predictions):
    logger.info("######## FINAL PREDICTIONS ##############")
    logger.info(f"Connecting to Orchestrator at http://{ORCHESTRATOR_URL}")

    # format predictions with the correct json output
    json_obj = formatted_predictions
    logger.info(f"Sending predictions to Orchestrator: {json_obj}")

    # send predictions with post API 
    url = f"http://{ORCHESTRATOR_URL}"

    ##
    response = requests.post(url, json=json_obj)
    logger.info("Data sent successfully to Orchestrator.")
    logger.info("Post predictions completed")
    ##

    # try:
    #     response = requests.post(url, json=json_obj)
        
    #     # This will raise an HTTPError if the status is not 2xx
    #     # It includes the status code and the reason (e.g., 404 Not Found)
    #     response.raise_for_status()
        
    #     logger.info("Data sent successfully to Orchestrator.")
    #     logger.info(f"Response from Orchestrator\nStatus Code: {response.status_code}\nResponse Text: {response.text}")
    #     logger.info("Post predictions completed")
        
    # except requests.exceptions.HTTPError as e:
    #     # Logs the specific error code and text before the task fails
    #     error_msg = f"API Error: {e.response.status_code} - {e.response.text}"
    #     logger.error(error_msg)
    #     # Re-raising the error tells Prefect to mark the task as FAILED
    #     raise 
    
    # except Exception as e:
    #     logger.error(f"Connection failed: {str(e)}")
    #     raise

    logger.info("#########################################")
    return


@flow
def ml_pipeline(metric_name: str = "cpu_usage_upf", model_name: str = "LSTM_cpu_usage_prometheus", target_name: str = "cpu_usage", node="6g-ntn-f5gc-w2", frequency: str = "5T", steps: int = 1):
    global logger
    logger = get_run_logger()   # initialize once per flow run

    # device check
    force_float32 = device_check()

    # load data
    historical_data = load_data(metric_name)

    # preprocessing data
    future_data_transformed = preprocessing.submit(historical_data, frequency)

    # model training
    series_transformed, transformer = future_data_transformed.result()
    future_my_model = model_training_production.submit(model_name, series_transformed, force_float32)
    my_model = future_my_model.result()

    # no need for evaluation in production
    
    # predict
    predictions = inference.submit(my_model, target_name, transformer, steps).result()

    # save predictions in postgres
    load_to_postgres.submit(predictions)

    # final format predictions
    formatted_predictions = final_format.submit(predictions, metric_name, target_name, node).result()

    # save predictions as artifact
    predictions = save_predictions_as_artifact.submit(formatted_predictions).result()

    # final format and send predictions
    post_predictions.submit(formatted_predictions)


if __name__ == "__main__":
    ml_pipeline()
