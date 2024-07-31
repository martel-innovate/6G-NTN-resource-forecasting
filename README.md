# Resource Forecasting in 6G Non-Terrestrial Networks (6G-NTN)


## Table of Contents

- [📖 Introduction](#introduction)
- [⚙️ Installation](#installation)
- [🏛️ Architecture](#architecture)
- [📉 Sequence Diagram](#sequence-diagram)
- [🏗️ Prefect Distributed Architecture](#prefect-distributed-architecture)
- [📜 License](#license)
- [📌 Acknowledgement](#acknowledgement)

## 📖Introduction

Welcome to 6G-NTN-ml project! This project leverages Machine Learning techniques to forecast resource allocation in the context of 6G Non-Terrestrial Networks (6G-NTN). Our goal is to develop ML solutions for resource forecasting, thereby enabling dynamic orchestration of virtual resources in 6G-NTN environments.

## ⚙Installation

### Prerequisites:

- Python 3.8+
- PostgreSQL
- Prometheus
- Docker 24.0.7

### Setup

1. Clone the repository:
```
git clone https://github.com/martel-innovate/6G-NTN-ml
cd 6G-NTN-ml
```

2. Run Docker desktop

3. Start docker compose
```
cd docker-compose
docker-compose --profile compose-project up -d --build
```

## 🏛Architecture

The AI-Powered Network Forecasting platform is essential for executing Machine Learning (ML) pipelines, enabling the automated training and retraining of ML models. ML and Deep Learning (DL) experiments are initiated by Prefect, a Machine Learning Function Orchestrator that orchestrates workflows utilizing a cloud-native infrastructure. The execution environment of Prefect flows is managed by Kubernetes. 

In the picture below, you can see the system architecture of our platform.

![Architecture of AI-Powered Network Forecasting](img/6g-ntn-architecture3.drawio.png)

## 📉Sequence Diagram

Below is the sequence diagram illustrating the workflow of data collection, storage, and prediction in our project:

![Sequence Diagram](img/sequence-diagram1.drawio.png)

## 🏗Prefect distributed architecture
Below is the architecture we used to interconnect Prefect components in a distributed environment. All components are deployed using Docker containers to ensure consistent and isolated execution environments.
![Prefect Architecture](img/prefect-architecture.drawio.png)

## 📜License

This project is licensed under the GPL3.0 License. See the [LICENSE](LICENSE) file for more details.

## 📌Acknowledgement

This repository is part of the **6G-NTN** project. 6G-NTN project has received funding from the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union’s Horizon Europe research and innovation programme under Grant Agreement No 101096479. This work has received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them. For more details about the project, visit the [6G-NTN project website](https://www.6g-ntn.eu/) or the [6G-NTN LinkedIn page](https://www.linkedin.com/company/6g-ntn/).


<img src="img/EUflagCoFunded6G-SNS_rgb_horizontal_negative.png" alt="European Union 6G SNS funding" width="30%"> <img src="img/WBF_SBFI_EU_Frameworkprogramme_E_RGB_neg_quer.png" alt="SERI" width="30%">
