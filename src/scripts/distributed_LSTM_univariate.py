from prefect import flow
import pandas as pd
import os
import requests
from prefect import flow, task
from prefect.artifacts import create_table_artifact, create_markdown_artifact
import logging

import pytz
from datetime import datetime

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
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
ENVIRONMENT = os.getenv('ENVIRONMENT')
ORCHESTRATOR_URL = os.getenv('ORCHESTRATOR_URL')

@task(log_prints=True)
def load_data(logger, metric_name):
    logger.info(f"PORT: {DB_PORT}")
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_SECRET}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}')
    conn = engine.connect()
    query = text(f'SELECT *  FROM input WHERE datetime::date < \'2024-07-15 \' AND metric_name LIKE \'{metric_name}%\'')
    results = conn.execute(query)
    results_list = results.fetchall()
    df = pd.DataFrame(results_list)
    return df

@task
def data_transformation(df): 
    # create one column for each metric
    df_pivot = df.pivot_table(index='datetime', columns='metric_name', values='value')

    # Preprocessing
    frequency = '1h'
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
    return series, train, test

@task
def normalize_series(series, train, test):
    # Normalize the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)
    series_transformed = transformer.transform(series)
    return series_transformed, train_transformed, test_transformed

@task(log_prints=True)
def preprocessing(df):
    print("Starting preprocessing")
    # data transformation
    df_resampled = data_transformation(df)
    # train/test split
    series, train, test = split_dataset(df_resampled)
    # data normalization
    series_transformed, train_transformed, val_transformed = normalize_series(series, train, test)

    print("Preprocessing completed")
    data_transformed = {'series' : series_transformed, 'train': train_transformed, 'val': val_transformed}
    return data_transformed

@task
def evaluation(model_name, model, series_transformed, val_transformed):
    # define train and test size
    train_size = int(0.7 * len(series_transformed))
    test_size = len(series_transformed) - train_size
    # predict
    pred_series = model.predict(n=test_size - 1)
    # eval
    mape_score = mape(pred_series, val_transformed)
    print(f"MAPE score: {mape_score}")

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

@task(log_prints=True)
def model_training(model_name, series_transformed, train_transformed, val_transformed):
    print("Starting model training")
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
        hidden_dim=20,
        dropout=0,
        batch_size=8,
        n_epochs=3, # TODO to be tuned
        optimizer_kwargs={"lr": 1e-3},
        model_name=model_name,
        log_tensorboard=True,
        random_state=42,
        training_length=20,
        input_chunk_length=5,
        output_chunk_length=1,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    # train model
    my_model.fit(train_transformed, val_series=val_transformed, verbose=True)

    # pick best model
    best_model = RNNModel.load_from_checkpoint(model_name=model_name, best=True)
    
    print("Model training completed")
    return best_model

@task(log_prints=True)
def inference(my_model, target_name):
    # model save_predictions
    print("Starting model save_predictions")
    predictions = my_model.predict(n=1)

    # create dataframes
    predictions_df = predictions.pd_dataframe()

    # set indexes
    predictions_df_new = predictions_df.reset_index()
    predictions_df_new.index = [target_name]
    return predictions_df_new

@task(log_prints=True)
def load_to_postgres(predictions):
    # Melt the DataFrame
    df_melted = pd.melt(predictions, id_vars=['datetime'], value_vars=predictions.columns.drop(["datetime"]), var_name='metric_name', value_name='value')

    # Drop the 'Unnamed: 0' column if necessary
    df_melted.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # Round the 'value' column to 4 decimal places
    df_melted['value'] = df_melted['value'].round(4)

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
        print("Data inserted successfully.")

    except SQLAlchemyError as e:
        # Rollback the transaction in case of error
        session.rollback()
        print(f"An error occurred: {e}")



@task(log_prints=True)
def save_predictions(predictions_cpu, predictions_memory):
    # load to postgres
    load_to_postgres(predictions_cpu)
    load_to_postgres(predictions_memory)

    # merge dataframes
    final_predictions = pd.concat([predictions_cpu, predictions_memory])
    
    # add datetime field
    final_predictions["datetime"] = final_predictions["datetime"].dt.tz_localize("Europe/Rome")  
    final_predictions["datetime"] = final_predictions["datetime"].apply(lambda x: x.isoformat())
    print(f"Final predictions:\n{final_predictions}") 
    
    # save predictions to Prefect Artifacts
    create_table_artifact(
        key="predictions",
        table=final_predictions.reset_index().to_dict(orient='records'),
        description="The output of the Machine Learning models for cpu and memory usage"
    ) 

    print("Model save_predictions completed")
    return  final_predictions

@task(log_prints=True)
def final_format(final_predictions):
    # Convert JSON data to pandas DataFrame
    df = pd.DataFrame(final_predictions)

    # Split the DataFrame into CPU and memory usage
    cpu_df = df[['datetime', 'cpu_usage_amf', 'cpu_usage_pcf', 'cpu_usage_smf', 'cpu_usage_usf']].copy()
    memory_df = df[['datetime', 'memory_usage_amf', 'memory_usage_pcf', 'memory_usage_smf', 'memory_usage_usf']].copy()

    # Rename columns for consistency with the desired output
    cpu_df.columns = ['datetime', 'amf', 'pcf', 'smf', 'usf']
    memory_df.columns = ['datetime', 'amf', 'pcf', 'smf', 'usf']

    # Add the index column
    cpu_df['index'] = 'cpu_usage'
    memory_df['index'] = 'memory_usage'

    # Drop rows where all amf, pcf, smf, usf are NaN
    cpu_df.dropna(subset=['amf', 'pcf', 'smf', 'usf'], how='all', inplace=True)
    memory_df.dropna(subset=['amf', 'pcf', 'smf', 'usf'], how='all', inplace=True)

    # Convert the DataFrames to dictionaries
    cpu_dict = cpu_df.to_json(orient='records')
    memory_dict = memory_df.to_json(orient='records')

    # Combine the dictionaries
    result = cpu_dict + memory_dict
    return result

@task(log_prints=True)
def post_predictions(logger, final_predictions):
    logger.info("Starting post predictions")

    # format predictions with the correct json output
    formatted_predictions = final_format.submit(final_predictions)

    # extract json
    json = formatted_predictions.result()

    # send predictions with post API 
    url = f"http://{ORCHESTRATOR_URL}/post"
    response = requests.post(url, json=json)

    # extract fields from response
    response_json = response.json()
    response_data = response_json["json"]
    response_origin = response_json["origin"]
    response_url = response_json["url"]

    logger.info("######## FINAL PREDICTIONS ##############")
    logger.info("Predictions has been sent to the server successfully!")
    logger.info(f"json: {response_data}")
    logger.info(f"from: {response_origin}")
    logger.info(f"To: {response_url}")
    logger.info("#########################################")
    logger.info("Post predictions completed")
    return

@flow(log_prints=True)
def ml_pipeline():
    # get logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load data
    dfs_cpu = load_data(logger, 'cpu')
    dfs_memory = load_data(logger, 'memory')

    # preprocessing data
    future_data_transformed_cpu = preprocessing.submit(dfs_cpu)
    future_data_transformed_memory = preprocessing.submit(dfs_memory)

    # first version
    # model training
    data_transformed_cpu = future_data_transformed_cpu.result()
    future_my_model_cpu = model_training.submit("LSTM_cpu_usage_prometheus", data_transformed_cpu['series'], data_transformed_cpu['train'], data_transformed_cpu['val'])
    my_model_cpu = future_my_model_cpu.result()
    
    data_transformed_memory = future_data_transformed_memory.result()
    future_my_model_memory = model_training.submit("LSTM_memory_usage_prometheus", data_transformed_memory['series'], data_transformed_memory['train'], data_transformed_memory['val'])
    my_model_memory = future_my_model_memory.result()

    # eval model
    evaluation("LSTM_cpu_usage_prometheus", my_model_cpu, data_transformed_cpu['series'], data_transformed_cpu['val'])
    evaluation("LSTM_memory_usage_prometheus", my_model_memory, data_transformed_memory['series'], data_transformed_memory['val'])
    
    # predict
    future_predictions_cpu = inference.submit(my_model_cpu, "cpu_usage")
    future_predictions_memory = inference.submit(my_model_memory, "memory_usage")

    final_predictions = save_predictions(future_predictions_cpu.result(), future_predictions_memory.result())

    # post predictions
    post_predictions(logger, final_predictions)


if __name__ == "__main__":
        ml_pipeline()
