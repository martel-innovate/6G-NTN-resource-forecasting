from prefect.filesystems import RemoteFileSystem
from dotenv import load_dotenv
import os
import boto3

# load env variables
load_dotenv()

MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_DEFAULT_BUCKETS = os.getenv('MINIO_DEFAULT_BUCKETS')
MINIO_HOST = os.getenv('MINIO_HOST')
MINIO_PORT = os.getenv('MINIO_PORT')

# Connect to MinIO
s3 = boto3.resource(
    's3',
    endpoint_url=f"http://{MINIO_HOST}:{MINIO_PORT}",
    aws_access_key_id=MINIO_ROOT_USER,
    aws_secret_access_key=MINIO_ROOT_PASSWORD,
)

# Create the bucket if it doesn't exist
for bucket in MINIO_DEFAULT_BUCKETS.split(','):
    bucket = bucket.strip()
    if not s3.Bucket(bucket) in s3.buckets.all():
        s3.create_bucket(Bucket=bucket)
        print(f"Created bucket: {bucket}")


minio_block = RemoteFileSystem(
    basepath=f"s3://{MINIO_DEFAULT_BUCKETS}",
    key_type="hash",
    settings=dict(
        use_ssl=False,
        key=MINIO_ROOT_USER,
        secret=MINIO_ROOT_PASSWORD,
        client_kwargs=dict(endpoint_url=f"http://{MINIO_HOST}:{MINIO_PORT}")
    ),
)
minio_block.save("minio", overwrite=True)
