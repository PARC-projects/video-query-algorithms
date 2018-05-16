"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import os
import time
import threading
from datetime import datetime
import logging
from api.api_repository import APIRepository
from models.compute_matches import compute_matches

LOOP_EXECUTION_TIME = 10.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))
FORMAT = '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s'
BASE_URL = "http://127.0.0.1:8000/"
default_weights = {'rgb': 1.0, 'warped_optical_flow': 1.5}
default_threshold = 0.8
streams = ('rgb', 'warped optical flow')

logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_NAME),
              logging.StreamHandler()])


def main():
    # Execute long pooling loop
    try:
        # check for updates
        query_update = APIRepository(BASE_URL)
        compute_matches(query_update, BASE_URL, default_weights, default_threshold, streams)
    except Exception as e:
        logging.error(e)
    finally:
        if os.environ.get('BROKER_THREADING') == 'True':
            threading.Timer(LOOP_EXECUTION_TIME, main).start()
        else:
            time.sleep(LOOP_EXECUTION_TIME)
            main()


if __name__ == '__main__':
    main()
