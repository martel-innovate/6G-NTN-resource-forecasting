FROM prefecthq/prefect:2-python3.10
RUN pip install psycopg2-binary
RUN pip install minio
RUN pip install s3fs
RUN pip install python-dotenv
COPY prefect-cli/requirements.txt .
RUN pip3 install -r requirements.txt
COPY /.env /root/.env