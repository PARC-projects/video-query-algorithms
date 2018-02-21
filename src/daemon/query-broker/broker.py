"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
import status
from datetime import datetime

LOOP_EXECUTION_TIME = 10.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))


def main():
    '''Execute long pooling loop'''
    queryStatus = status.QueryStatus()
    threading.Timer(LOOP_EXECUTION_TIME, main).start()
    queryStatus.getStatus()


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        format=
        '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s',
        level=logging.DEBUG,
        handlers=[logging.FileHandler(LOG_NAME),
                  logging.StreamHandler()])
    main()
