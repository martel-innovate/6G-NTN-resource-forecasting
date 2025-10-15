FROM prefecthq/prefect:2-python3.10
RUN pip install psycopg2-binary
RUN pip install minio
RUN pip install s3fs
RUN pip install python-dotenv
RUN pip install boto3
COPY /.env /root/.env