from minio import Minio
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()

MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_DEFAULT_BUCKETS = os.getenv('MINIO_DEFAULT_BUCKETS')

MINIO_SETTINGS = {
    "key": MINIO_ROOT_USER,
    "secret": MINIO_ROOT_PASSWORD,
    "port": 9000,
    "host": "minio",
    "prefect-bucket": MINIO_DEFAULT_BUCKETS,
}

# endpoint = f"{MINIO_SETTINGS['host']}:{MINIO_SETTINGS['port']}"
endpoint = "minio:9000"  # put docker container instead of localhost
endpoint

# load client
client = Minio(
    endpoint=endpoint,
    access_key=MINIO_SETTINGS["key"],
    secret_key=MINIO_SETTINGS["secret"],
    secure=False,
)

# define variables
bucket_name = "prefect"
local_path = "./scripts"

# Iterate over all files in the directory
for filename in os.listdir(local_path):
    file_path = os.path.join(local_path, filename)

    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        print(f"added file named {filename}")
        # load file to Minio
        client.fput_object(
            bucket_name=bucket_name,
            object_name="scripts/" + filename,
            file_path=file_path,
            
        )

# define variables
bucket_name = "prefect"
local_path = "./scripts/prometheus_client"

# Iterate over all files in the directory
for filename in os.listdir(local_path):
    file_path = os.path.join(local_path, filename)

    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        print(f"added file named {filename}")
        # load file to Minio
        client.fput_object(
            bucket_name=bucket_name,
            object_name="scripts/prometheus_client/" + filename,
            file_path=file_path,
            
        )
