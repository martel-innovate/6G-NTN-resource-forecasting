FROM prefecthq/prefect:2-python3.10
RUN pip install psycopg2-binary
RUN pip install minio
RUN pip install s3fs
RUN apt-get update && apt-get install -y curl
COPY init_prefect.sh /init_prefect.sh
RUN chmod +x /init_prefect.sh
ENTRYPOINT ["/bin/bash", "-c", "prefect server start --host $PREFECT_ORION_API_HOST & /init_prefect.sh && wait"]