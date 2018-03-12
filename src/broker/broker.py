"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
import status
import os
import sys
from datetime import datetime
import logging

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir, 'models'))
import compute_matches

LOOP_EXECUTION_TIME = 10.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))
FORMAT = '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s'

logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(LOG_NAME),
        logging.StreamHandler()
    ]
)


def main():
    '''Execute long pooling loop'''
    queryStatus = status.QueryStatus()
    threading.Timer(LOOP_EXECUTION_TIME, main).start()
    try:
        result = queryStatus.getStatus()
        compute_matches.new_matches(result["compute_new_matches"])
        compute_matches.revised_matches(result["compute_similarity"], [])
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    main()
