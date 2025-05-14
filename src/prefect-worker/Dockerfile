FROM prefecthq/prefect:2-python3.10
RUN pip install --upgrade pip
RUN pip install psycopg2-binary
RUN pip install minio
RUN pip install s3fs
COPY requirements.txt .
RUN pip3 install -r requirements.txt