"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
from api.status import QueryStatus
from datetime import datetime
import logging
from models import compute_matches

LOOP_EXECUTION_TIME = 10.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))
FORMAT = '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s'

logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_NAME),
              logging.StreamHandler()])


def main():
    '''Execute long pooling loop'''
    queryStatus = QueryStatus()

    try:
        result = queryStatus.getStatus()
        if result["new"]:
            compute_matches.new_matches(result["new"])
        if result["revise"]:
            compute_matches.revised_matches(result["revise"], [])
    except Exception as e:
        logging.error(e)
    finally:
        # create a new thread
        threading.Timer(LOOP_EXECUTION_TIME, main).start()


if __name__ == '__main__':
    main()
