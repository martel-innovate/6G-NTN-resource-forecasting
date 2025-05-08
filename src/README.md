## Instructions for docker compose
### Setup components individually

1. Run Docker desktop

2. Start Prefect (Server + Database)
```
cd docker-compose
docker-compose --profile prefect-orion up -d --build
```
Dashboard is accessible at http://localhost:4200/dashboard
If needed, create new work pool from UI.

4. Start a Worker (pool name is defined in docker-compose file)
```
docker-compose --profile prefect-worker up -d --build
```

5. Start MinIO
```
docker-compose --profile minio up -d
```
UI of the storage is accessible at http://localhost:9001/login

6. Start and run Prefect CLI
```
docker-compose run --build prefect-cli
```
To deploy a pipeline from Prefect CLI:
```
# run script to set MinIO as block storage
python scripts/set_block_storage.py

# create yaml file ready for deploy
prefect deployment build scripts/<<sfile_name.py>>:<<function_name>> -n '<<deploy_name>>' --pool '<<pool_name>>' -sb 'remote-file-system/minio'

# real example: this creates a file in MinIO called 'ml_pipeline-deployment.yaml'
prefect deployment build scripts/distributed_LSTM_univariate.py:ml_pipeline -n 'ml_flow' --pool 'LSTM_forecasting' -sb 'remote-file-system/minio' 

# deploy pipeline
prefect deployment apply ml_pipeline-deployment.yaml  
```

6. Start PostgreSQL
```
docker-compose --profile postgres up -d
```

7. Start Httpbin orchestrator to accept requests
```
docker-compose --profile orchestrator up -d
``` 

8. Go to Prefect UI and execute Quick run