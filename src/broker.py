"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import os
import random
import threading
import logging
from datetime import datetime
from api.api_repository import APIRepository
from models.compute_matches import compute_matches
from models import Hyperparameter

###########################################
# Broker Config
###########################################
LOOP_EXECUTION_TIME = 5.0  # In seconds
BASE_URL = "http://127.0.0.1:8000/"

###########################################
# Logging Config
###########################################
LOG_NAME = 'logs/query_broker_{0}.log'.format(datetime.now().strftime("%Y_%m_%d"))
logging.basicConfig(
    format='%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s',
    level=logging.INFO,  # level=logging.DEBUG would print a lot more info
    handlers=[logging.FileHandler(LOG_NAME), logging.StreamHandler()]
)

###########################################
# Hyperparameter defaults
###########################################
default_weights = {
    'rgb': 1.0,
    'warped_optical_flow': 1.5
}
default_threshold = 0.8
near_miss_default = 0.5
streams = (
    'rgb',
    'warped_optical_flow'
)
feature_name = 'global_pool'
mu = 0.05
bootstrap_type = 'bagging'  # type of bootstrapping, one of 'simple', 'bagging', or 'partial_update'
nbags = 3
# f_bootstrap is the fraction of matches and invalid clips to use in bootstrapping. Using a value less than 1 is one
# way to reduce overfitting.
# The bootstrapped clips are adjusted for all streams and splits, so leaving some out of
# bootstrapping forces the ensemble averaging to do more work.
f_bootstrap = 1
# In target bootstrapping, a new target is averaged with the old one as f_memory*new + (1-f_memory)*old
f_memory = 0.7
# ballast should be >=0 and <1.
# False positives penalty reduced by (1-ballast), false negative penalty increased by (1+ballast)
ballast = 0.0


def main():
    try:
        # Ask data-source for queries to impart logic on.
        query_updates = APIRepository(BASE_URL)

        # Calculate hyperparameters
        hyperparameters = Hyperparameter(
            default_weights,
            default_threshold,
            ballast,
            near_miss_default,
            mu,
            streams,
            feature_name,
            f_bootstrap,
            f_memory,
            bootstrap_type,
            nbags
        )

        # If available, set random seed on enviroment to ease debugging
        if os.environ["RANDOM_SEED"] != "None":
            random.seed(a=os.environ["RANDOM_SEED"])

        # Compute new matches and scores for a query
        compute_matches(query_updates, hyperparameters)
    except Exception as e:
        logging.error(e, exc_info=True)
    finally:
        if os.environ.get('BROKER_THREADING') == 'True':
            threading.Timer(LOOP_EXECUTION_TIME, main).start()


if __name__ == '__main__':
    main()
