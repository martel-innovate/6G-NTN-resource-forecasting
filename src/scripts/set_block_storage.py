from prefect.filesystems import RemoteFileSystem
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()

MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_DEFAULT_BUCKETS = os.getenv('MINIO_DEFAULT_BUCKETS')

MINIO_SETTINGS = {
    'key': MINIO_ROOT_USER,
    'secret': MINIO_ROOT_PASSWORD,
    'port': 9000,
    'host': "minio",
    'prefect-bucket': MINIO_DEFAULT_BUCKETS
}

minio_block = RemoteFileSystem(
    basepath=f"s3://{MINIO_SETTINGS['prefect-bucket']}",
    key_type="hash",
    settings=dict(
        use_ssl=False,
        key=MINIO_SETTINGS['key'],
        secret=MINIO_SETTINGS['secret'],
        client_kwargs=dict(endpoint_url=f"http://{MINIO_SETTINGS['host']}:{MINIO_SETTINGS['port']}")
    ),
)
minio_block.save("minio", overwrite=True)
