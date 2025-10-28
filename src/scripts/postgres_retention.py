from prefect import flow, task, get_run_logger
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_NAME=os.getenv('DB_NAME')
DB_USER=os.getenv('DB_USER')
DB_PASSWORD=os.getenv('DB_SECRET')
DB_HOST=os.getenv('DB_HOSTNAME')
DB_PORT=os.getenv('DB_PORT')
PROMETHEUS_HOSTNAME = os.getenv('PROMETHEUS_HOSTNAME')

logger = None

@task
def delete_from_db(table: str, retention_days: int):
    """
    Deletes rows older than retention_days from the specified table.
    """
    global logger
    logger.info(f"Deleting records older than {retention_days} day(s) from table {table}.")
    logger.info(f"Connecting to database {DB_NAME} at {DB_HOST}:{DB_PORT} as user {DB_USER}")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        sql = f"DELETE FROM {table} WHERE datetime < NOW() - INTERVAL '{retention_days} day'"
        cursor.execute(sql)
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Deleted {deleted} rows from {table}.")
    except Exception as e:
        logger.error(f"Database error: {e}")

@flow
def cleanup_metrics(table: str, retention_days: int):
    """
    Prefect flow to clean up old data.
    Reads table name and retention_days from environment variables.
    """
    global logger
    logger = get_run_logger()   # initialize once per flow run
    delete_from_db(table, retention_days)

if __name__ == '__main__':
    cleanup_metrics()