FROM prefecthq/prefect:2-python3.10
ARG TARGETARCH
COPY requirements-${TARGETARCH}.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install psycopg2-binary
RUN pip install minio
RUN pip install s3fs
RUN pip3 install -r requirements.txt