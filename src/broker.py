"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import os
import threading
from datetime import datetime
import logging
from api.api_repository import APIRepository
from models.compute_matches import compute_matches
from models import Hyperparameter

LOOP_EXECUTION_TIME = 5.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))
FORMAT = '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s'
BASE_URL = "http://127.0.0.1:8000/"
# hyperparameter defaults
default_weights = {'rgb': 1.0, 'warped_optical_flow': 1.5}
default_threshold = 0.8
near_miss_default = 0.5
streams = ('rgb', 'warped_optical_flow')
feature_name = 'global_pool'
mu = 0.05
# ballast should be >=0 and <1.
# False positives penalty reduced by (1-ballast), false negative penalty increased by (1+ballast)
ballast = 0.3

logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,  # level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_NAME),
              logging.StreamHandler()])


def main():
    # Execute long pooling loop
    try:
        # check for updates
        query_updates = APIRepository(BASE_URL)
        hyperparameters = Hyperparameter(default_weights, default_threshold, ballast, near_miss_default, streams,
                                         feature_name, mu)
        compute_matches(query_updates, hyperparameters)
    except Exception as e:
        logging.error(e)
    finally:
        if os.environ.get('BROKER_THREADING') == 'True':
            threading.Timer(LOOP_EXECUTION_TIME, main).start()


if __name__ == '__main__':
    main()
