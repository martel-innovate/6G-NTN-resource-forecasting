import requests
import os
from dotenv import load_dotenv

# load env variables
load_dotenv()

PROMETHEUS_HOSTNAME = os.getenv('PROMETHEUS_HOSTNAME')
PROMETHEUS_PORT = os.getenv('PROMETHEUS_PORT')

class PrometheusClient(object):
    def __init__(
            self,
            prometheus_base_url=f"{PROMETHEUS_HOSTNAME}:{PROMETHEUS_PORT}",
            http_protocol='http'
    ):
        self.prometheus_base_url = prometheus_base_url
        self.http_protocol = http_protocol
    
    @staticmethod
    def http(
            url,
            method="GET",
            **kwargs
    ):
        response = requests.request(
            method=method,
            url=url,
            **kwargs
        )
        return response
    
    def get_metrics(
            self
    ):
        response = self.http(
            url=f'{self.http_protocol}://{self.prometheus_base_url}/api/v1/label/__name__/values'
        )
        return response
    
    def fetch_relevant_metrics(
            self, metrics_str_tuple:str
    ):
        response = self.get_metrics()
        if response.status_code == 200:
            response_json = response.json()
            all_metrics = response_json['data']
            relevant_metrics = list(filter(
                lambda metric: metric.startswith(metrics_str_tuple) is True,
                all_metrics
            ))
            return relevant_metrics
        else:
            return print("404")
        
    def intent_query(
            self,
            metric,
            time=None,
            time_range=None
    ):
        query_url = f'{self.http_protocol}://{self.prometheus_base_url}/api/v1/query?query={metric}'
        if time:
            query_url = f'{query_url}&time={time}'
        if time_range:
            query_url = f'{query_url}[{time_range}]'
            print(query_url)
        response = self.http(
            url=query_url
        )
        return response

    def range_query(
            self,
            metric,
            start,
            end
    ):
        range_url = f'{self.http_protocol}://{self.prometheus_base_url}/api/v1/query_range?query={metric}'
        if start:
            range_url = f'{range_url}&start={start}'
        if end:
            range_url = f'{range_url}&end={end}'
        response = self.http(
            url=range_url
        )
        return response