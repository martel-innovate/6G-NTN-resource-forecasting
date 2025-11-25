from prefect import flow
import pandas as pd
import os
import numpy as np
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

import pytz
from datetime import datetime

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
    #query = text(f'SELECT * FROM input WHERE datetime::date < \'2024-07-15 \' AND metric_name LIKE \'{metric_name}%\'')
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
def split_dataset(df):
    # define train and test size
    train_size = int(0.7 * len(df))
    test_size = len(df) - train_size
    split_point = df.iloc[train_size].name

    # create darts TimeSeries
    if "datetime" in df:
        series = TimeSeries.from_dataframe(df, "datetime")
    else:
        series = TimeSeries.from_dataframe(df.reset_index(), "datetime")

    # train test split
    train, test = series.split_after(split_point)
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return series, train, test

@task
def normalize_series(series, train, test):
    # Normalize the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)
    series_transformed = transformer.transform(series)
    return series_transformed, train_transformed, test_transformed, transformer

@task
def preprocessing(df, frequency):
    logger.info(f"Starting preprocessing, frequency={frequency}")
    # data transformation
    df_resampled = data_transformation(df, frequency)
    # train/test split
    series, train, test = split_dataset(df_resampled)
    # data normalization
    series_transformed, train_transformed, val_transformed, transformer = normalize_series(series, train, test)

    logger.info("Preprocessing completed")
    data_transformed = {'series' : series_transformed, 'train': train_transformed, 'val': val_transformed}
    return data_transformed, transformer

@task
def evaluation(model_name, model, series_transformed, val_transformed):
    # define train and test size
    train_size = int(0.7 * len(series_transformed))
    test_size = len(series_transformed) - train_size
    # predict
    pred_series = model.predict(n=test_size - 1)
    # eval
    mape_score = mape(pred_series, val_transformed)
    logger.info(f"MAPE score: {mape_score}")

    markdown = f""" 
    ### Evaluation
    This is the evaluation score for the model "{model_name}": 
    - MAPE --> {mape_score}
    """
    create_markdown_artifact(
        key="mape-score",
        markdown=markdown,
        description="MAPE score"
    )
    return

@task
def model_training(model_name, series_transformed, train_transformed, val_transformed, force_float32):
    logger.info("Starting model training")

    if force_float32:  
        train_transformed = train_transformed.astype(np.float32)
        val_transformed = val_transformed.astype(np.float32)
        logger.info("Default torch dtype set to float32")

    # define early stopping parameters
    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.0005,
        mode="min",
    )

    pl_trainer_kwargs = {"callbacks": [my_stopper]}

    # build model
    my_model = RNNModel(
        model="LSTM",
        hidden_dim=50,
        dropout=0.1,
        batch_size=8,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-3},
        model_name=model_name,
        log_tensorboard=True,
        random_state=42,
        training_length=48,
        input_chunk_length=6,
        output_chunk_length=1,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    # train model
    my_model.fit(train_transformed, val_series=val_transformed, verbose=True)

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


@flow
def ml_pipeline(metric_name: str = "cpu_usage_upf", model_name: str = "LSTM_cpu_usage_prometheus", target_name: str = "cpu_usage", frequency: str = "5m", steps: int = 1):
    global logger
    logger = get_run_logger()   # initialize once per flow run

    # device check
    force_float32 = device_check()

    # load data
    historical_data = load_data(metric_name)

    # preprocessing data
    future_data_transformed = preprocessing.submit(historical_data, frequency)

    # model training
    data_transformed, transformer = future_data_transformed.result()
    future_my_model = model_training.submit(model_name, data_transformed['series'], data_transformed['train'], data_transformed['val'], force_float32)
    my_model = future_my_model.result()

    # eval model
    evaluation(model_name, my_model, data_transformed['series'], data_transformed['val'])
    
    # predict
    future_predictions = inference.submit(my_model, target_name, transformer, steps)

    # save predictions in postgres
    load_to_postgres.submit(future_predictions.result())


if __name__ == "__main__":
    ml_pipeline()
