# Notebooks - 6G-NTN Resource Forecasting Repository

Welcome to the notebooks folder of the 6G-NTN Resource Forecasting repository. This folder contains various Jupyter notebooks developed to explore the data used in the project and to experiment with different forecasting models and techniques. The notebooks also include visualizations to help interpret the results effectively.

![LSTM multivariate forecasting](../img/lstm-multivariate-forecasting.png)

## Contents

### 1. Data Analysis

- **Notebook**: `data_analysis.ipynb`
- **Description**: This notebook provides a comprehensive analysis of the dataset used in the project. It includes data cleaning, exploration, and visualization to understand the underlying patterns and relationships in the data.
- **Environment**: `environment_datascience.yml`

### 2. BASELINES

- **Notebook CPU**: `baseline_cpu.ipynb`
- **Notebook memory**: `baseline_memory.ipynb`
- **Description**: These notebooks showcases the implementation of a baseline model. It includes the selection of a simple model, and it's usage a benchmark to compare the performance of more complex models.
- **Environment**: `environment_darts.yml`

### 3. ARIMA Model

- **Notebook CPU**: `ARIMA_cpu.ipynb`
- **Notebook memory**: `ARIMA_memory.ipynb`
- **Description**: These notebooks demonstrate how to implement and train an ARIMA (AutoRegressive Integrated Moving Average) model. It includes model selection, fitting, and forecasting steps, along with the visualization of the results.
- **Environment**: `environment_datascience.yml`

### 4. GRU Model

- **Notebook CPU**: `GRU_cpu.ipynb`
- **Notebook memory**: `GRU_memory.ipynb`
- **Description**: These notebooks show how to implement and train a GRU (Gated Recurrent Unit) model. It includes model selection, fitting, and forecasting steps, along with the visualization of the results.
- **Environment**: `environment_darts.yml`

### 5. LSTM Model using TensorFlow

- **Notebook**: `LSTM_model.ipynb`
- **Description**: This notebook showcases the development of LSTM (Long Short-Term Memory) models using TensorFlow. It covers:
  - **Vanilla LSTM**: A basic implementation of an LSTM model.
  - **Complex LSTM**: An enhanced version of the LSTM model with additional layers and configurations for improved performance.
- **Environment**: `environment_tensorflow.yml`

### 6. LSTM Model using Darts

- **Notebooks CPU**: `LSTM_univariate_cpu.ipynb` and `LSTM_multivariate_cpu.ipynb`
- **Notebooks memory**: `LSTM_univariate_memory.ipynb` and `LSTM_multivariate_memory.ipynb`
- **Description**: These notebooks demonstrate the use of the Darts library to implement LSTM models for forecasting. It includes:
  - **Univariate Forecasting**: Training an LSTM model to forecast on time series using one variable (itself).
  - **Multivariate Forecasting**: Training an LSTM model to forecast multiple time series simultaneously.
  - **LSTM with Covariates**: Using additional covariate data to improve the forecasting accuracy of the primary time series.
- **Environment**: `environment_darts.yml`

## How to Use

1. **Clone the Repository**: Ensure you have cloned the repository to your local machine.
2. **Navigate to the root Folder**: Change your directory to the `root` folder.
3. **Activate the preferred environment**: Activate the environment using Conda
4. **Open the Notebooks**: Use Jupyter Notebook or JupyterLab to open and run the notebooks.
   ```
   jupyter notebook
   ```

## How to Use Environments

To create and activate a new Conda environment based on the `environment_xxx.yml` file located in the environments folder, follow these steps:

1. Download and install Conda from [here](https://docs.conda.io/en/latest/miniconda.html).
2. Open a terminal and navigate to the root folder of the project.
3. Create the environment using the provided `environment_xxx.yml` file. For example, to create an environment with Tensorflow:
    ```bash
    conda env create -f notebooks/environment_tensorflow.yml
    ```
4. Activate the newly created environment:
    ```bash
    conda activate tensorflow
    ```
5. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```


## Access to Dataset

For these notebooks, we used the `amf-performance.csv` file developed in the context of [this publication](https://www.eurecom.fr/publication/6971).
The CSV file, as well as the entire dataset, can be accessed from [Zenodo](https://zenodo.org/records/6907619).

To run the notebooks, you must download the `amf-performance.csv` file and place it in the `data` directory.

### Citation
Mohamed Mekki, Nassima Toumi, & Adlen Ksentini. (2022). *Benchmarking on Microservices Configurations and the Impact on the Performance in Cloud Native Environments* (1.0) [Data set]. LCN 2022, 47th Annual IEEE Conference on Local Computer Networks, Edmonton, Canada. Zenodo. [https://doi.org/10.5281/zenodo.6907619](https://doi.org/10.5281/zenodo.6907619)

